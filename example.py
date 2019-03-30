# -*- coding: utf-8 -*-

from parser import BiaffineParser, Model
from parser.utils import TextDataset, collate_fn
import torch
from torch.utils.data import DataLoader


BATCH_SIZE = 32				# only affects speed, if too big you could OOM
CHECKPOINT_DIR = 'model.pt' # you'll need to change this
VOCAB_DIR = 'vocab.pt'		# and this

vocab = torch.load(VOCAB_DIR)

bert = BertModel.from_pretrained('bert-base-multilingual-cased')

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
network = BiaffineParser(params)			  # if you want to use the original (not tuned) BERT
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

example(sentences)