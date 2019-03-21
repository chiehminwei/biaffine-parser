# -*- coding: utf-8 -*-

from parser.modules import CHAR_LSTM, LSTM, MLP, Biaffine
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from pytorch_pretrained_bert import BertTokenizer, BertModel
import subprocess
import os

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

    def __init__(self, params, embeddings):
        super(BiaffineParser, self).__init__()

        self.params = params
        # # the embedding layer
        # self.pretrained = nn.Embedding.from_pretrained(embeddings)
        # self.embed = nn.Embedding(num_embeddings=params['n_words'],
        #                           embedding_dim=params['n_embed'])
        # # the char-lstm layer
        # self.char_lstm = CHAR_LSTM(n_chars=params['n_chars'],
        #                            n_embed=params['n_char_embed'],
        #                            n_out=params['n_char_out'])
        self.embed_dropout = IndependentDropout(p=params['embed_dropout'])

        # BERT
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        # the word-lstm layer
        # self.lstm = LSTM(input_size=params['n_embed']+params['n_char_out'],
        #                  hidden_size=params['n_lstm_hidden'],
        #                  num_layers=params['n_lstm_layers'],
        #                  dropout=params['lstm_dropout'],
        #                  bidirectional=True)
        self.lstm_dropout = SharedDropout(p=params['lstm_dropout'])

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['mlp_dropout'])
        self.mlp_arc_d = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['mlp_dropout'])
        self.mlp_rel_h = MLP(n_in=params['n_lstm_hidden']*2,
                             n_hidden=params['n_mlp_rel'],
                             dropout=params['mlp_dropout'])
        self.mlp_rel_d = MLP(n_in=params['n_lstm_hidden']*2,
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
        self.pad_index = params['pad_index']
        self.unk_index = params['unk_index']

        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.zeros_(self.embed.weight)

    def forward(self, words, mask):
        # run words through bert

        # get the mask and lengths of given batch
        lens = words.ne(self.pad_index).sum(dim=1)
        # mask = words.ne(self.pad_index)
        # lens = mask.sum(dim=1)
        
        # get outputs from embedding layers
        embed, _ = self.bert(words, attention_mask=mask, output_all_encoded_layers=False)
        embed, char_embed = self.embed_dropout(embed, embed)
        x = embed

        #print('attention_mask', mask.shape)
        #print(mask[0])
        #print('bert_output', x.shape)
        #print(x[0, :10])

        # embed = self.pretrained(words)
        # embed += self.embed(
        #     words.masked_fill_(words.ge(self.embed.num_embeddings),
        #                        self.unk_index)
        # )
        # char_embed = self.char_lstm(chars[mask])
        # char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)
        # embed, char_embed = self.embed_dropout(embed, char_embed)
        # # concatenate the word and char representations
        # x = torch.cat((embed, char_embed), dim=-1)
        

        # sorted_lens, indices = torch.sort(lens, descending=True)
        # inverse_indices = indices.argsort()
        # x = pack_padded_sequence(x[indices], sorted_lens, True)
        # x = self.lstm(x)
        # x, _ = pad_packed_sequence(x, True)
        # x = self.lstm_dropout(x)[inverse_indices]
        
        # apply MLPs to the LSTM output states
        arc_h = self.mlp_arc_h(x)
        #print('arch_h', arc_h.shape)
        #print(arc_h[0, :10])
        arc_d = self.mlp_arc_d(x)
        #print('arc_d', arc_d.shape)
        #print(arc_d[0, :10])
        rel_h = self.mlp_rel_h(x)
        #print('rel_h', rel_h.shape)
        #print(rel_h[0, :10])
        rel_d = self.mlp_rel_d(x)
        #print('rel_d', rel_d.shape)
        #print(rel_d[0, :10])

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        #print('s_arc', s_arc.shape)
        #print(s_arc[0, :10])
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        #print('s_rel', s_rel.shape)
        #print(s_rel[0, :10])
        # set the scores that exceed the length of each sentence to -inf
        len_mask = length_to_mask(lens, max_len=words.shape[-1], dtype=torch.uint8)
        s_arc.masked_fill_((1 - len_mask).unsqueeze(1), float('-inf'))
        #print('s_arc_after_mask', s_arc.shape)
        #print(s_arc[0, :10])
        return s_arc, s_rel

    @classmethod
    def load(cls, fname, cloud_address):
        # Copy from cloud if there's no saved checkpoint
        if not os.path.isfile(fname):
            FNULL = open(os.devnull, 'w')
            subprocess.call(['gsutil', 'cp', cloud_address+fname, fname], stdout=FNULL, stderr=subprocess.STDOUT)
        # Proceed only if either [1] copy success [2] local file already exists
        if os.path.isfile(fname):
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            state = torch.load(fname, map_location=device)
            network = cls(state['params'], state['embeddings'])
            network.load_state_dict(state['state_dict'])
            network.to(device)

        return network

    def save(self, fname, epoch, cloud_address):
        state = {
            'params': self.params,
            'embeddings': self.pretrained.weight,
            'state_dict': self.state_dict(),
            'last_epoch': epoch,
        }
        torch.save(state, fname)
        # Save a copy to cloud as well
        FNULL = open(os.devnull, 'w')
        subprocess.call(['gsutil', 'cp', fname, cloud_address+fname], stdout=FNULL, stderr=subprocess.STDOUT)
