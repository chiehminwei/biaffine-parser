# -*- coding: utf-8 -*-

from parser import BiaffineParser, Model
from parser.utils import TextDataset, collate_fn

import torch
from torch.utils.data import DataLoader


BATCH_SIZE = 32
CHECKPOINT_DIR = 'model.pt'
VOCAB_DIR = 'vocab.pt'

vocab = torch.load(VOCAB_DIR)
network = BiaffineParser.load(CHECKPOINT_DIR)
model = Model(vocab, network)


def example(sentences):

	dataset = TextDataset(vocab.numericalize_sentences(sentences))
	loader = DataLoader(dataset=dataset,
	                    batch_size=BATCH_SIZE,
	                    collate_fn=collate_fn)

	# Three options
	embeddings = model.get_embeddings(loader)
	s_arc, s_rel = model.get_matrices(loader)
	s_arc, s_rel, embeddings = model.get_everything(loader)