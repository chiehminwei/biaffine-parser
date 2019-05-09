# -*- coding: utf-8 -*-

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from parser.modules import (MLP, Biaffine, BiLSTM, IndependentDropout,
                            SharedDropout)

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import subprocess
import os
import logging


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class BiaffineParser(nn.Module):

    def __init__(self, params):
        super(BiaffineParser, self).__init__()

        self.params = params
        
        # BERT
        lm = BertForMaskedLM.from_pretrained(params['bert_model'])
        self.bert = lm.bert
        self.cls = lm.cls
        self.config = lm.config
        self.bert_down_projection = MLP(n_in=params['n_bert_hidden'],
                                        n_hidden=params['n_embed'])
        self.bert_dropout = SharedDropout(p=params['bert_dropout'])
        

        self.tag_embed = None
        if params['use_pos']:
            self.tag_embed = nn.Embedding(num_embeddings=params['n_tags'],
                                          embedding_dim=params['n_tag_embed'])
            nn.init.zeros_(self.tag_embed.weight)

        # LSTM layer
        self.lstm = None
        if params['use_lstm']:
            for param in self.bert.parameters():
                param.requires_grad = False
            if params['use_pos']:
                self.lstm = BiLSTM(input_size=params['n_embed'] + params['n_tag_embed'],
                             hidden_size=params['n_lstm_hidden'],
                             num_layers=params['n_lstm_layers'],
                             dropout=params['lstm_dropout'])
            else:
                self.lstm = BiLSTM(input_size=params['n_embed'],
                                 hidden_size=params['n_lstm_hidden'],
                                 num_layers=params['n_lstm_layers'],
                                 dropout=params['lstm_dropout'])
            self.lstm_dropout = SharedDropout(p=params['lstm_dropout'])

            self.mlp_arc_h = MLP(n_in=params['n_lstm_hidden'] * 2,
                                 n_hidden=params['n_mlp_arc'],
                                 dropout=params['mlp_dropout'])
            self.mlp_arc_d = MLP(n_in=params['n_lstm_hidden'] * 2,
                                 n_hidden=params['n_mlp_arc'],
                                 dropout=params['mlp_dropout'])
            self.mlp_rel_h = MLP(n_in=params['n_lstm_hidden'] * 2,
                                 n_hidden=params['n_mlp_rel'],
                                 dropout=params['mlp_dropout'])
            self.mlp_rel_d = MLP(n_in=params['n_lstm_hidden'] * 2,
                                 n_hidden=params['n_mlp_rel'],
                                 dropout=params['mlp_dropout'])

        # the MLP layers
        else:
            if params['use_pos']:
                self.mlp_arc_h = MLP(n_in=params['n_embed'] + params['n_tag_embed'],
                                     n_hidden=params['n_mlp_arc'],
                                     dropout=params['mlp_dropout'])
                self.mlp_arc_d = MLP(n_in=params['n_embed'] + params['n_tag_embed'],
                                     n_hidden=params['n_mlp_arc'],
                                     dropout=params['mlp_dropout'])
                self.mlp_rel_h = MLP(n_in=params['n_embed'] + params['n_tag_embed'],
                                     n_hidden=params['n_mlp_rel'],
                                     dropout=params['mlp_dropout'])
                self.mlp_rel_d = MLP(n_in=params['n_embed'] + params['n_tag_embed'],
                                     n_hidden=params['n_mlp_rel'],
                                     dropout=params['mlp_dropout'])

            else:
                self.mlp_arc_h = MLP(n_in=params['n_embed'],
                                     n_hidden=params['n_mlp_arc'],
                                     dropout=params['mlp_dropout'])
                self.mlp_arc_d = MLP(n_in=params['n_embed'],
                                     n_hidden=params['n_mlp_arc'],
                                     dropout=params['mlp_dropout'])
                self.mlp_rel_h = MLP(n_in=params['n_embed'],
                                     n_hidden=params['n_mlp_rel'],
                                     dropout=params['mlp_dropout'])
                self.mlp_rel_d = MLP(n_in=params['n_embed'],
                                     n_hidden=params['n_mlp_rel'],
                                     dropout=params['mlp_dropout'])

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=params['n_mlp_arc'],
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=params['n_mlp_rel'],
                                 n_out=params['n_rels'],
                                 bias_x=True,
                                 bias_y=True)
        
    def forward(self, input_ids, mask, masked_lm_labels=None, tags=None):
        # get the mask and lengths of given batch
        lens = mask.sum(dim=1)
 
        # get outputs from bert
        # if self.lstm:
        #     # Use BERT 9-12
        #     layers = []
        #     bert_output, _ = self.bert(input_ids, attention_mask=mask)
        #     for layer in range(8, 12):
        #         layers.append(bert_output[layer])
        #     sequence_output = torch.sum(torch.stack(layers), dim=0)
        # else:
        #     sequence_output, _ = self.bert(input_ids, attention_mask=mask, output_all_encoded_layers=False)
        # del _
        sequence_output = input_ids

        sequence_output = self.bert_down_projection(sequence_output)

        if tags is not None:
            tag_embed = self.tag_embed(tags)
            sequence_output= torch.cat((sequence_output, tag_embed), dim=-1)

        # Dependency parsing
        x = sequence_output

        # bert dropout 
        x = self.bert_dropout(x)

        # LSTM
        if self.lstm:
            sorted_lens, indices = torch.sort(lens, descending=True)
            inverse_indices = indices.argsort()
            x = pack_padded_sequence(x[indices], sorted_lens, True)
            x = self.lstm(x)
            x, _ = pad_packed_sequence(x, True)
            x = self.lstm_dropout(x)[inverse_indices]

        # apply MLPs to the BERT output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        # set the scores that exceed the length of each sentence to -inf
        len_mask = length_to_mask(lens, max_len=input_ids.shape[-1], dtype=torch.uint8)
        s_arc.masked_fill_((1 - len_mask).unsqueeze(1), float('-inf'))

        if masked_lm_labels is None:
            return s_arc, s_rel
        else: # Masked LM
            prediction_scores = self.cls(sequence_output)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return s_arc, s_rel, masked_lm_loss

    def get_embeddings(self, words, mask, layer_index=-1, return_all=False):
        # get outputs from bert
        encoded_layers, _ = self.bert(words, attention_mask=mask, output_all_encoded_layers=True)
        del _
        if return_all:
            return encoded_layers
        else:
            return encoded_layers[layer_index]

    def get_concat_embeddings(self, words, mask):
        # get outputs from bert
        x, _ = self.bert(words, attention_mask=mask, output_all_encoded_layers=False)
        del _
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        return torch.cat((arc_h, arc_d, rel_h, rel_d), -1)

    def get_everything(self, words, mask, layer_index=-1, return_all=False):
        # get the mask and lengths of given batch
        lens = mask.sum(dim=1)
        
        # get outputs from bert
        encoded_layers, _ = self.bert(words, attention_mask=mask, output_all_encoded_layers=True)
        del _
        if return_all:
            embed_to_return = encoded_layers
        else:
            embed_to_return = encoded_layers[:,layer_index]
        
        embed, _ = self.bert(words, attention_mask=mask, output_all_encoded_layers=False)
        del _
        x = embed

        # bert dropout
        x = self.bert_dropout(x)

        # apply MLPs to the BERT output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        # set the scores that exceed the length of each sentence to -inf
        len_mask = length_to_mask(lens, max_len=words.shape[-1], dtype=torch.uint8)
        s_arc.masked_fill_((1 - len_mask).unsqueeze(1), float('-inf'))

        return s_arc, s_rel, embed

    @classmethod
    def load(cls, fname, cloud_address=None, local_rank=0):
        # Copy from cloud if there's no saved checkpoint
        if not os.path.isfile(fname):
            if cloud_address:
                FNULL = open(os.devnull, 'w')
                cloud_address = os.path.join(cloud_address, fname)
                # subprocess.call(['gsutil', 'cp', cloud_address, fname], stdout=FNULL, stderr=subprocess.STDOUT)
        # Proceed only if either [1] copy success [2] local file already exists
        if os.path.isfile(fname):
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            state = torch.load(fname, map_location='cpu')
            network = cls(state['params'])
            network.load_state_dict(state['state_dict'])
            network.to(device)
            logging.info('Loaded model from checkpoint (local rank {})'.format(local_rank))
        else:
            raise IOError('Local checkpoint does not exists. Failed to load model.')

        return network

    def save(self, fname, epoch, cloud_address, optimizer, max_metric, local_rank=0, is_best=False):
        state = {
            'params': self.params,
            'state_dict': self.state_dict(),
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'max_metric': max_metric,
        }
        torch.save(state, fname)
        if is_best:
            logging.info("Best model saved.")
        else:
            logging.info("Latest model saved.")

        # Save a copy to cloud as well
        # FNULL = open(os.devnull, 'w')
        # cloud_address = os.path.join(cloud_address, fname)
        # subprocess.call(['gsutil', 'cp', fname, cloud_address], stdout=FNULL, stderr=subprocess.STDOUT)
