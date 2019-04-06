# -*- coding: utf-8 -*-

from config import Config
from parser import BiaffineParser, Model
from parser.utils import Corpus, TextDataset, Vocab, collate_fn
import torch
from torch.utils.data import DataLoader


BATCH_SIZE = 32				# only affects speed, if too big you could OOM
CHECKPOINT_DIR = 'model.pt' # you'll need to change this
VOCAB_DIR = 'vocab.pt'		# and this

vocab = torch.load(VOCAB_DIR)

# what's in the params won't affect embeddings, they're just here so that my initialization code doesn't break
params = {
	'n_words': vocab.n_train_words,
	'n_chars': vocab.n_chars,
	'word_dropout': Config.word_dropout,
	'n_bert_hidden': Config.n_bert_hidden,
	'bert_dropout': Config.bert_dropout,
	'n_mlp_arc': Config.n_mlp_arc,
	'n_mlp_rel': Config.n_mlp_rel,
	'mlp_dropout': Config.mlp_dropout,
	'n_rels': vocab.n_rels,
	'pad_index': vocab.pad_index
}
# network = BiaffineParser(params)			  # if you want to use the original (not tuned) BERT
network = BiaffineParser.load(CHECKPOINT_DIR) # if you want to use the tuned BERT
model = Model(vocab, network)

sentences = [['Yes', 'yes', 'yes', '.'], ["It's", 'all', 'done', ':)']]

def example(sentences):

	dataset = TextDataset(vocab.numericalize_sentences(sentences))
	loader = DataLoader(dataset=dataset,
	                    batch_size=BATCH_SIZE,
	                    collate_fn=collate_fn)

	# Three options
	embeddings = model.get_embeddings(loader)
	s_arc, s_rel = model.get_matrices(loader)
	s_arc, s_rel, embeddings = model.get_everything(loader)

# example(sentences)

def PennTreebank(corpus_path, out_file, meta_file):
	corpus = Corpus.load(corpus_path)
	vocab = Vocab.from_corpus(corpus=corpus, min_freq=2)
	dataset = TextDataset(vocab.numericalize_tags(corpus))
	loader = DataLoader(dataset=dataset,
	                    batch_size=BATCH_SIZE,
	                    collate_fn=collate_fn)
	embeddings = model.get_embeddings(loader)
	with open(out_file, 'w') as f, open(meta_file, 'w') as ff:
		for sentence in embeddings:
			for word_embed in sentence:
				f.write('\t'.join([str(val) for val in word_embed])+'\n')
		ff.write('Word\tPOS\n')
		for words, tags in zip(corpus.words, corpus.tags):
			for word, tag in zip(words, tags):
				ff.write(word + '\t' + tag + '\n')

PennTreebank('data/dev.conllx', 'embeddings.tsv', 'meta.tsv')
