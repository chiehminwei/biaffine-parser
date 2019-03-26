# -*- coding: utf-8 -*-

from parser.modules import MLP, Biaffine
from parser.modules.dropout import IndependentDropout, SharedDropout
from pytorch_pretrained_bert import BertTokenizer, BertModel

import torch
import torch.nn as nn

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

    def __init__(self, params):
        super(BiaffineParser, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.params = params
        self.embed_dropout = nn.Dropout(p=params['embed_dropout'])
        # self.embed_dropout = IndependentDropout(p=params['embed_dropout'])

        # BERT
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert_dropout = SharedDropout(p=params['bert_dropout'])

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=params['n_bert_hidden'],
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['mlp_dropout'])
        self.mlp_arc_d = MLP(n_in=params['n_bert_hidden'],
                             n_hidden=params['n_mlp_arc'],
                             dropout=params['mlp_dropout'])
        self.mlp_rel_h = MLP(n_in=params['n_bert_hidden'],
                             n_hidden=params['n_mlp_rel'],
                             dropout=params['mlp_dropout'])
        self.mlp_rel_d = MLP(n_in=params['n_bert_hidden'],
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
        
    def forward(self, words, mask):
        # get the mask and lengths of given batch
        lens = words.ne(self.pad_index).sum(dim=1)
        # word dropout
        words = self.embed_dropout(words.type('torch.DoubleTensor')).type('torch.LongTensor').to(self.device)
        
        # get outputs from bert
        embed, _ = self.bert(words, attention_mask=mask, output_all_encoded_layers=False)
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

        return s_arc, s_rel

    @classmethod
    def load(cls, fname, cloud_address):
        # Copy from cloud if there's no saved checkpoint
        if not os.path.isfile(fname):
            FNULL = open(os.devnull, 'w')
            cloud_address = os.path.join(cloud_address, fname)
            subprocess.call(['gsutil', 'cp', cloud_address, fname], stdout=FNULL, stderr=subprocess.STDOUT)
        # Proceed only if either [1] copy success [2] local file already exists
        if os.path.isfile(fname):
            state = torch.load(fname, map_location=self.device)
            network = cls(state['params'])
            network.load_state_dict(state['state_dict'])
            network.to(device)
            print('Loaded model from checkpoint')
        else:
            raise IOError('Local checkpoint does not exists. Failed to load model.')

        return network

    def save(self, fname, epoch, cloud_address):
        state = {
            'params': self.params,
            # 'embeddings': self.pretrained.weight,
            'state_dict': self.state_dict(),
            'last_epoch': epoch,
        }
        torch.save(state, fname)
        # Save a copy to cloud as well
        FNULL = open(os.devnull, 'w')
        cloud_address = os.path.join(cloud_address, fname)
        subprocess.call(['gsutil', 'cp', fname, cloud_address], stdout=FNULL, stderr=subprocess.STDOUT)
