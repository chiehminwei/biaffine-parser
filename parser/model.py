# -*- coding: utf-8 -*-

from parser.metric import AttachmentMethod
from parser.parser import BiaffineParser
from parser.utils import PregeneratedDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert import BertAdam, BertTokenizer
import numpy as np

from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import logging

import torch.optim.lr_scheduler


class Model(object):
    def __init__(self, vocab, network):
        super(Model, self).__init__()
        self.vocab = vocab
        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        
    def __call__(self, loaders, epochs, num_data_epochs, patience, lr, t_total, last_epoch, 
                 cloud_address, args, batch_size, gradient_accumulation_steps=1, max_metric=0.0):

        train_dataloader, dev_loader, test_loader = loaders
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        total_time = timedelta()
        max_e, max_metric = last_epoch, max_metric

        # Prepare optimizer
        param_optimizer = list(self.network.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.train_lm:
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=lr,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)
        else:
            self.optimizer = BertAdam(optimizer_grouped_parameters, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max',
            factor=args.step_decay_factor,
            patience=args.step_decay_patience,
            verbose=True,
        )

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        if args.local_rank == 0:
            logging.info('***Started training at {}***'.format(datetime.now()))            
        
        for epoch in range(last_epoch + 1, epochs + 1):
            start = datetime.now()
            if args.train_lm:
                epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                                num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
                if args.distributed:
                    train_sampler = DistributedSampler(epoch_dataset)
                else:
                    train_sampler = RandomSampler(epoch_dataset)
                
                train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=batch_size//gradient_accumulation_steps)
            
            stats = {'tr_loss': 0, 'lm_loss': 0, 'arc_loss': 0, 'rel_loss': 0, 'nb_tr_examples': 0, 'nb_tr_steps': 0}
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
                self.train(train_dataloader, pbar, stats, args, data_parallel=bool(torch.cuda.device_count() > 1 and not args.no_cuda and not args.distributed))
            
            train_loss, train_metric = self.evaluate(train_dataloader, trainset=args.train_lm)
            dev_loss, dev_metric = self.evaluate(dev_loader)
            self.scheduler.step(dev_metric)
            test_loss, test_metric = self.evaluate(dev_loader)
            t = datetime.now() - start
            total_time += t            
            if args.local_rank == 0:
                logging.info(f"{'train:':<6} Loss: {train_loss:.4f} {train_metric}")
                logging.info(f"{'dev:':<6} Loss: {dev_loss:.4f} {dev_metric}")
                logging.info(f"{'test:':<6} Loss: {test_loss:.4f} {test_metric}")         
                logging.info(f"{t}s elapsed\n")
            
            if args.local_rank == 0:
                model_to_save = self.network.module if hasattr(self.network, 'module') else self.network  # Only save the model itself
                if epoch % 2 == 0: # Save latest every two epochs
                    output_model_file = args.checkpoint_dir / "model_epoch{}.pt".format(epoch)
                    model_to_save.save(output_model_file, epoch, cloud_address, self.optimizer, dev_metric)

                if dev_metric > max_metric: # Save best
                    output_model_file = args.checkpoint_dir / "model_best.pt"
                    max_e, max_metric = epoch, dev_metric
                    model_to_save.save(output_model_file, epoch, cloud_address, self.optimizer, max_metric, is_best=True)

                elif epoch - max_e >= patience: # Early stopping
                    break

        if args.local_rank == 0:
            logging.info('***Finished training at {}***'.format(datetime.now()))
            logging.info(f"max score of dev is {max_metric.score:.2%} at epoch {max_e}")
            logging.info(f"mean time of each epoch is {total_time / epoch}s")
            logging.info(f"{total_time}s elapsed")

    def train(self, loader, pbar, stats, args, data_parallel=False):
        self.network.train()
        for step, batch in enumerate(loader):
            batch = tuple(t.to(self.device) for t in batch)

            if args.train_lm:
                input_ids, arc_ids, rel_ids, input_masks, word_start_masks, word_end_masks, lm_label_ids = batch 
                s_arc, s_rel, lm_loss = self.network(input_ids, input_masks, masked_lm_labels=lm_label_ids)
                lm_loss = torch.mean(lm_loss)
            elif args.use_pos:
                input_ids, input_masks, word_start_masks, arc_ids, rel_ids, tag_ids = batch
                s_arc, s_rel = self.network(input_ids, input_masks, tags=tag_ids)
            else:
                input_ids, input_masks, word_start_masks, arc_ids, rel_ids, tag_ids = batch
                s_arc, s_rel = self.network(input_ids, input_masks)
            
            word_start_masks[:, 0] = 0  # ignore [CLS]
            lens = input_masks.sum(dim=1) - 1 # ignore [SEP]
            word_start_masks[torch.arange(len(word_start_masks)), lens] = 0
            
            gold_arcs, gold_rels = arc_ids[word_start_masks], rel_ids[word_start_masks]
            s_arc, s_rel = s_arc[word_start_masks], s_rel[word_start_masks]            

            # Get loss
            arc_loss, rel_loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            loss = arc_loss + rel_loss
            if args.train_lm:
                loss += lm_loss

            if data_parallel:
                loss = loss.mean() # mean() to average on multi-gpu.
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Handle tqdm
            stats['tr_loss'] += loss.item()
            if args.train_lm:
                stats['lm_loss'] += lm_loss.item()
            stats['arc_loss'] += arc_loss.item()
            stats['rel_loss'] += rel_loss.item()
            stats['nb_tr_examples'] += input_ids.size(0)
            stats['nb_tr_steps'] += 1
            pbar.update(1)
            mean_loss = stats['tr_loss'] * self.gradient_accumulation_steps / stats['nb_tr_steps']
            pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")

            # Step optimizer
            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
        mean_loss = stats['tr_loss'] * self.gradient_accumulation_steps / stats['nb_tr_steps']
        mean_lm_loss = stats['lm_loss'] * self.gradient_accumulation_steps / stats['nb_tr_steps']
        mean_arc_loss = stats['arc_loss'] * self.gradient_accumulation_steps / stats['nb_tr_steps']
        mean_rel_loss = stats['rel_loss'] * self.gradient_accumulation_steps / stats['nb_tr_steps']
        if args.local_rank == 0:
            if args.train_lm:
                logging.info(f"{'train:':<6} Loss: {mean_loss:.4f} Arc: {mean_arc_loss:.4f} Rel: {mean_rel_loss:.4f} LM: {mean_lm_loss:.4f}")
            else:
                logging.info(f"{'train:':<6} Loss: {mean_loss:.4f} Arc: {mean_arc_loss:.4f} Rel: {mean_rel_loss:.4f}")
        
    @torch.no_grad()
    def evaluate(self, loader, include_punct=False, trainset=False):
        self.network.eval()

        loss, metric = 0, AttachmentMethod()
        for i, batch in enumerate(loader):
            batch = tuple(t.to(self.device) for t in batch)
            
            if trainset:
                input_ids, arc_ids, rel_ids, input_masks, word_start_masks, word_end_masks, lm_label_ids = batch 
            else: 
                input_ids, input_masks, word_start_masks, arc_ids, rel_ids = batch
            
            # ignore [CLS]
            word_start_masks[:, 0] = 0
            # ignore [SEP]
            lens = input_masks.sum(dim=1) - 1
            word_start_masks[torch.arange(len(word_start_masks)), lens] = 0

            # ignore all punctuation if specified 
            if not include_punct:
                puncts = input_ids.new_tensor([punct for punct in self.vocab.puncts])
                word_start_masks &= input_ids.unsqueeze(-1).ne(puncts).all(-1)

            s_arc, s_rel = self.network(input_ids, input_masks)
            s_arc, s_rel = s_arc[word_start_masks], s_rel[word_start_masks]
            
            gold_arcs, gold_rels = arc_ids[word_start_masks], rel_ids[word_start_masks]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)
            
            arc_loss, rel_loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            loss += arc_loss + rel_loss
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

        loss /= len(loader)

        return loss, metric

    # Not fixed yet don't use
    @torch.no_grad()
    def predict(self, loader):
        self.network.eval()

        all_arcs, all_rels = [], []
        for words, attention_mask, token_start_mask, arcs, rels in loader:
            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = attention_mask.sum(dim=1) - 1
            token_start_mask[torch.arange(len(token_start_mask)), lens] = 0

            s_arc, s_rel = self.network(words, attention_mask)
            s_arc, s_rel = s_arc[token_start_mask], s_rel[token_start_mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            # lens for splitting
            lens = token_start_mask.sum(dim=1).tolist()

            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        # loss = arc_loss + rel_loss

        return arc_loss, rel_loss

    def decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels

    @torch.no_grad()
    def get_embeddings(self, loader, layer_index=-1, return_all=False, ignore=True, ignore_token_start_mask=False):
        self.network.eval()

        all_embeddings = []
        for words, attention_mask, token_start_mask in loader:
            if ignore_token_start_mask:
                token_start_mask = attention_mask.clone()
 
            if ignore:
                # ignore [CLS]
                token_start_mask[:, 0] = 0
                # ignore [SEP]
                lens = attention_mask.sum(dim=1) - 1
                token_start_mask[torch.arange(len(token_start_mask)), lens] = 0
                
            embed = self.network.get_embeddings(words, attention_mask, layer_index, return_all=return_all)
            
            if return_all:
                embed = torch.stack(embed)        # [num_layer, batch_size, seq_len, bert_dim]
                embed = embed[:,token_start_mask] # [num_layer, num_word, bert_dim]
            else:
                embed = embed[token_start_mask]   # [num_word, bert_dim]      
            
            # lens for splitting
            lens = token_start_mask.sum(dim=1).tolist()
            for sentence_embed in torch.split(embed, lens, dim=-2):
                all_embeddings.append(np.array(sentence_embed.tolist()))
            
        return all_embeddings

    @torch.no_grad()
    def get_avg_embeddings(self, loader, ignore=True, layer_index=-1):
        self.network.eval()

        all_embeddings = []
        for words, attention_mask, token_start_mask in loader:
            # [batch_size, seq_len, bert_dim]
            embed = self.network.get_embeddings(words, attention_mask, layer_index)
            
            if ignore:
                # ignore [CLS]
                token_start_mask[:, 0] = 0
                # ignore [SEP]
                lens = attention_mask.sum(dim=1) - 1
                token_start_mask[torch.arange(len(token_start_mask)), lens] = 0
                # need to take care of attention as well since we later rely on attention to do averaging
                attention_mask[torch.arange(len(token_start_mask)), lens] = 0

            for sent_embed, sent_att_mask, sent_mask in zip(embed, attention_mask, token_start_mask):
                sent_avg_embeddings = []
                tmp = None
                tmp_len = 0
                sent_embed = sent_embed.tolist()
                sent_att_mask = sent_att_mask.tolist()
                sent_mask = sent_mask.tolist()
                for word_embed, word_att_mask, word_mask in zip(sent_embed, sent_att_mask, sent_mask):
                    if word_att_mask != 1:
                        if tmp is not None:
                            sent_avg_embeddings.append(tmp/tmp_len)
                        tmp = None
                        break
                    if word_mask == 1:
                        if tmp is not None:
                            if tmp_len == 0:
                                tmp_len = 1
                            sent_avg_embeddings.append(tmp/tmp_len)
                        tmp = np.array(word_embed)
                        tmp_len = 1
                    else:
                        if tmp is not None:
                            tmp += np.array(word_embed)
                            tmp_len += 1

                # take care of last word when sentence len == max_seq_len in batch
                if tmp is not None:
                    sent_avg_embeddings.append(tmp/tmp_len)

                all_embeddings.append(np.array(sent_avg_embeddings))

        return all_embeddings

    @torch.no_grad()
    def get_concat_embeddings(self, loader):
        self.network.eval()

        all_embeddings = []
        for words, attention_mask, token_start_mask in loader:    
            embed = self.network.get_concat_embeddings(words, attention_mask)
            embed = embed[token_start_mask]   # [num_word, bert_dim]      
            
            # lens for splitting
            lens = token_start_mask.sum(dim=1).tolist()
            for sentence_embed in torch.split(embed, lens, dim=-2):
                all_embeddings.append(np.array(sentence_embed.tolist()))
            
        return all_embeddings

    @torch.no_grad()
    def get_avg_concat_embeddings(self, loader, ignore=True, layer_index=-1):
        self.network.eval()

        all_embeddings = []
        for words, attention_mask, token_start_mask in loader:
            # [batch_size, seq_len, bert_dim]
            embed = self.network.get_concat_embeddings(words, attention_mask)
            
            if ignore:
                # ignore [CLS]
                token_start_mask[:, 0] = 0
                # ignore [SEP]
                lens = attention_mask.sum(dim=1) - 1
                token_start_mask[torch.arange(len(token_start_mask)), lens] = 0
                # need to take care of attention as well since we later rely on attention to do averaging
                attention_mask[torch.arange(len(token_start_mask)), lens] = 0

            for sent_embed, sent_att_mask, sent_mask in zip(embed, attention_mask, token_start_mask):
                sent_avg_embeddings = []
                tmp = None
                tmp_len = 0
                sent_embed = sent_embed.tolist()
                sent_att_mask = sent_att_mask.tolist()
                sent_mask = sent_mask.tolist()
                for word_embed, word_att_mask, word_mask in zip(sent_embed, sent_att_mask, sent_mask):
                    if word_att_mask != 1:
                        if tmp is not None:
                            sent_avg_embeddings.append(tmp/tmp_len)
                        tmp = None
                        break
                    if word_mask == 1:
                        if tmp is not None:
                            if tmp_len == 0:
                                tmp_len = 1
                            sent_avg_embeddings.append(tmp/tmp_len)
                        tmp = np.array(word_embed)
                        tmp_len = 1
                    else:
                        if tmp is not None:
                            tmp += np.array(word_embed)
                            tmp_len += 1

                # take care of last word when sentence len == max_seq_len in batch
                if tmp is not None:
                    sent_avg_embeddings.append(tmp/tmp_len)

                all_embeddings.append(np.array(sent_avg_embeddings))

        return all_embeddings         

    @torch.no_grad()
    def get_everything(self, loader):
        self.network.eval()

        all_arcs, all_rels, all_embeddings = [], [], []
        for words, attention_mask, token_start_mask in loader:
            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = attention_mask.sum(dim=1) - 1
            token_start_mask[torch.arange(len(token_start_mask)), lens] = 0

            s_arc, s_rel, embed = self.network.get_everything(words, attention_mask)
            s_arc, s_rel, embed = s_arc[token_start_mask], s_rel[token_start_mask], embed[token_start_mask]
            
            # lens for splitting
            lens = token_start_mask.sum(dim=1).tolist()
            for i, sentence_arc in enumerate(torch.split(s_arc, lens)):
                all_arcs.append(np.array(sentence_arc[:,:lens[i]].tolist()))

            for i, sentence_rel in enumerate(torch.split(s_rel, lens)):
                all_rels.append(np.array(sentence_rel[:,:lens[i]].tolist()))

            for sentence_embed in torch.split(embed, lens, dim=-2):
                all_embeddings.append(np.array(sentence_embed.tolist()))

        return all_arcs, all_rels, all_embeddings

    @torch.no_grad()
    def get_matrices(self, loader):
        self.network.eval()

        all_arcs, all_rels = [], []
        for words, attention_mask, token_start_mask in loader:
            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = attention_mask.sum(dim=1) - 1
            token_start_mask[torch.arange(len(token_start_mask)), lens] = 0

            s_arc, s_rel = self.network(words, attention_mask)
            s_arc, s_rel = s_arc[token_start_mask], s_rel[token_start_mask]
            
            # lens for splitting
            lens = token_start_mask.sum(dim=1).tolist()
            for i, sentence_arc in enumerate(torch.split(s_arc, lens)):
                all_arcs.append(np.array(sentence_arc[:,:lens[i]].tolist()))

            for i, sentence_rel in enumerate(torch.split(s_rel, lens)):
                all_rels.append(np.array(sentence_rel[:,:lens[i]].tolist()))

        return all_arcs, all_rels
