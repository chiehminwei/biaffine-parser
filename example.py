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

sentences = [['I', 'luv', 'Chloe', '.'], ['But', 'maybe', 'not', 'anymore', ':)']]

def example(sentences):

	dataset = TextDataset(vocab.numericalize_sentences(sentences))
	loader = DataLoader(dataset=dataset,
	                    batch_size=BATCH_SIZE,
	                    collate_fn=collate_fn)

	# Three options
	print('embeddings')
	embeddings = model.get_embeddings(loader)
	print(len(embeddings))
	print(len(embeddings[0]))
	print(len(embeddings[1]))
	print(len(embeddings[0][0]))

	print('s_arc, s_rel')
	s_arc, s_rel = model.get_matrices(loader)
	print(len(s_arc), len(s_rel))
	print(len(s_arc[0]), len(s_rel[0]))
	print(len(s_arc[1]), len(s_rel[1]))
	print(len(s_arc[0][0]), len(s_rel[0][0]))
	print(len(s_arc[0][0]), len(s_rel[0][0][0]))

	print('embeddings')
	s_arc, s_rel, embeddings = model.get_everything(loader)
	print(len(embeddings))
	print(len(embeddings[0]))
	print(len(embeddings[1]))
	print(len(embeddings[0][0]))

	print('s_arc, s_rel')
	print(len(s_arc), len(s_rel))
	print(len(s_arc[0]), len(s_rel[0]))
	print(len(s_arc[1]), len(s_rel[1]))
	print(len(s_arc[0][0]), len(s_rel[0][0]))

example(sentences)