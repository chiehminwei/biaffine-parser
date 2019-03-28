# -*- coding: utf-8 -*-

from parser.metric import AttachmentMethod
from parser.parser import BiaffineParser

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert import BertTokenizer

from datetime import datetime, timedelta
from tqdm import tqdm

import numpy as np

class Model(object):

    def __init__(self, vocab, network):
        super(Model, self).__init__()

        self.vocab = vocab
        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __call__(self, loaders, epochs, patience,
                 lr, betas, epsilon, weight_decay, annealing, file,
                 last_epoch, cloud_address, gradient_accumulation_steps=1):

        self.gradient_accumulation_steps = gradient_accumulation_steps
        total_time = timedelta()
        max_e, max_metric = 0, 0.0
        train_loader, dev_loader, test_loader = loaders
        self.optimizer = BertAdam(params=self.network.parameters(),
                                  lr=lr, b1=betas[0], b2=betas[1],
                                  e=epsilon, weight_decay=weight_decay,
                                  max_grad_norm=5.0)
        # self.optimizer = optim.Adam(params=self.network.parameters(),
        #                             lr=lr, betas=betas, eps=epsilon)
        # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
        #                                              lr_lambda=annealing)
        print('***Started training at {}***'.format(datetime.now()))
        for epoch in range(last_epoch + 1, epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            # self.train(train_loader)
            embeddings = np.array(self.get_embeddings(train_loader))
            s_arc, s_rel = self.get_matrix(train_loader)
            s_arc = np.array(s_arc)
            s_rel = np.array(s_rel)
            print(embeddings.shape)
            print(s_arc.shape)
            print(s_rel.shape)
            assert 1 == 2

            print(f"Epoch {epoch} / {epochs}:")
            loss, train_metric = self.evaluate(train_loader)
            print(f"{'train:':<6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self.evaluate(dev_loader)
            print(f"{'dev:':<6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self.evaluate(test_loader)
            print(f"{'test:':<6} Loss: {loss:.4f} {test_metric}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # save the model if it is the best so far
            if dev_metric > max_metric:
                if not torch.cuda.device_count() > 1:
                    self.network.save(file, epoch, cloud_address)
                else:
                    self.network.module.save(file, epoch, cloud_address)
                max_e, max_metric = epoch, dev_metric
            elif epoch - max_e >= patience:
                break
        print('***Finished training at {}***'.format(datetime.now()))
        self.network = BiaffineParser.load(file, cloud_address)
        loss, metric = self.evaluate(test_loader)

        print(f"max score of dev is {max_metric.score:.2%} at epoch {max_e}")
        print(f"the score of test at epoch {max_e} is {metric.score:.2%}")
        print(f"mean time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def train(self, loader):
        self.network.train()
        for step, batch in enumerate(tqdm(loader)):
            batch = tuple(t.to(self.device) for t in batch)
            words, attention_mask, token_start_mask, arcs, rels = batch

            s_arc, s_rel = self.network(words, attention_mask)
            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = words.ne(self.vocab.pad_index).sum(dim=1) - 1
            token_start_mask[torch.arange(len(token_start_mask)), lens] = 0

            s_arc, s_rel = s_arc[token_start_mask], s_rel[token_start_mask]
            gold_arcs, gold_rels = arcs[token_start_mask], rels[token_start_mask]

            loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                self.optimizer.step()
                # self.scheduler.step()
                self.optimizer.zero_grad()

            # print(self.tokenizer.convert_ids_to_tokens(words[token_start_mask].detach().to(torch.device("cpu")).numpy()))
            # for sentence in words:
            #     print(self.tokenizer.convert_ids_to_tokens(sentence.detach().to(torch.device("cpu")).numpy()))

    @torch.no_grad()
    def evaluate(self, loader, include_punct=False):
        self.network.eval()

        loss, metric = 0, AttachmentMethod()
        for i, batch in enumerate(loader):
            batch = tuple(t.to(self.device) for t in batch)
            words, attention_mask, token_start_mask, arcs, rels = batch

            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = words.ne(self.vocab.pad_index).sum(dim=1) - 1
            token_start_mask[torch.arange(len(token_start_mask)), lens] = 0

            # ignore all punctuation if specified
            if not include_punct:
                puncts = words.new_tensor([punct for punct in self.vocab.puncts])
                token_start_mask &= words.unsqueeze(-1).ne(puncts).all(-1)

            s_arc, s_rel = self.network(words, attention_mask)
            s_arc, s_rel = s_arc[token_start_mask], s_rel[token_start_mask]
            gold_arcs, gold_rels = arcs[token_start_mask], rels[token_start_mask]
            # try:
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)
            # except:
            #     print(i)
            #     print(words.size()[0])
            #     for sentence in words:
            #         print(self.tokenizer.convert_ids_to_tokens(sentence.detach().to(torch.device("cpu")).numpy()))

            loss += self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.network.eval()

        all_arcs, all_rels = [], []
        for words, attention_mask, token_start_mask, arcs, rels in tqdm(loader):
            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = words.ne(self.vocab.pad_index).sum(dim=1) - 1
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

    @torch.no_grad()
    def get_embeddings(self, loader):
        self.network.eval()

        all_embeddings = []
        for words, attention_mask, token_start_mask, arcs, rels in tqdm(loader):
            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = words.ne(self.vocab.pad_index).sum(dim=1) - 1
            token_start_mask[torch.arange(len(token_start_mask)), lens] = 0

            embed = self.network.get_embeddings(words, attention_mask)
            print('')
            print('Original network embedding shape')
            print(embed.shape)
            embed = embed[token_start_mask]
            print('Original network after masking embedding shape')
            print(embed.shape)

            # lens for splitting
            lens = token_start_mask.sum(dim=1).tolist()
            all_embeddings.extend(torch.split(embed, lens))

            print('Splitting turns embedding into:')

            for yeet in torch.split(embed, lens):
                print(yeet.shape)
            
        all_embeddings = [seq.tolist() for seq in all_embeddings]
        
        # to numpy
        return all_embeddings

    @torch.no_grad()
    def get_matrix(self, loader):
        self.network.eval()

        all_arcs, all_rels = [], []
        for words, attention_mask, token_start_mask, arcs, rels in tqdm(loader):
            # ignore [CLS]
            token_start_mask[:, 0] = 0
            # ignore [SEP]
            lens = words.ne(self.vocab.pad_index).sum(dim=1) - 1
            token_start_mask[torch.arange(len(token_start_mask)), lens] = 0

            s_arc, s_rel = self.network(words, attention_mask)
            s_arc, s_rel = s_arc[token_start_mask], s_rel[token_start_mask]
            
            # lens for splitting
            lens = token_start_mask.sum(dim=1).tolist()

            all_arcs.extend(torch.split(s_arc, lens))
            all_rels.extend(torch.split(s_rel, lens))

        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [seq.tolist()  for seq in all_rels]

        # to numpy
        return all_arcs, all_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels
