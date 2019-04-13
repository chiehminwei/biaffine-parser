# -*- coding: utf-8 -*-

from config import Config
from parser import BiaffineParser, Model
from parser.utils import Corpus, TextDataset, Vocab, collate_fn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import h5py
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer


BATCH_SIZE = 8				# only affects speed, if too big you could OOM
CHECKPOINT_DIR = 'model.pt' # path to model checkpoint
VOCAB_DIR = 'vocab.pt'		# path to vocab checkpoint

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
network = BiaffineParser(params)			  # if you want to use the original (not tuned) BERT
syntactic_network = BiaffineParser.load(CHECKPOINT_DIR) # if you want to use the tuned BERT

if torch.cuda.is_available():
	network.to(torch.device('cuda'))
	# syntactic_network.to(torch.device('cuda'))

model = Model(vocab, network)
syntactic_model = Model(vocab, syntactic_network)

sentences = [['Yes', 'yes', 'yes', '.'], ["It's", 'all', 'done', ':)', '.']]


def example(sentences):

	dataset = TextDataset(vocab.numericalize_sentences(sentences))
	loader = DataLoader(dataset=dataset,
						batch_size=BATCH_SIZE,
						collate_fn=collate_fn)

	# set ignore=True to not return embeddings for start of sentence and end of sentence tokens
	# set return_all=True to return embeddings for all 12 layers, return_all=False to return only last layer
	# default is ignore=True, return_all=False
	
	embeddings = model.get_embeddings(loader, ignore=True, return_all=False, ignore_token_start_mask=True)
	avg_embeddings = model.get_avg_embeddings(loader)

	tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
	print(tokenizer.tokenize(' '.join(sentences[0])))
	print(embeddings[0].shape)
	print(avg_embeddings[0].shape)
	print(embeddings[0][:,:3])
	print(avg_embeddings[0][:,:3])

	print(tokenizer.tokenize(' '.join(sentences[1])))
	print(embeddings[1].shape)
	print(avg_embeddings[1].shape)
	print(embeddings[1][:,:3])
	print(avg_embeddings[1][:,:3])

	

	# s_arc, s_rel = model.get_matrices(loader)
	
	# If you get embeddings this way, it is default (ignore=True, return_all=False)
	# s_arc, s_rel, embeddings = model.get_everything(loader)

# This function is for embedding visualization
def PennTreebank(corpus_path, out_file, meta_file):
	corpus = Corpus.load(corpus_path)
	vocab = Vocab.from_corpus(corpus=corpus, min_freq=2)
	a, b, c, words, tags = vocab.numericalize_tags(corpus)
	dataset = TextDataset((a, b, c))
	loader = DataLoader(dataset=dataset,
						batch_size=BATCH_SIZE,
						collate_fn=collate_fn)
	original_embeddings = model.get_embeddings(loader)
	syntactic_embeddings = syntactic_model.get_embeddings(loader)
	with open(out_file, 'w') as f, open(meta_file, 'w') as ff:
		embeddings = []
		embeddings2 = []
		for sentence in tqdm(original_embeddings):
			for word_embed in sentence:
				embeddings.append(torch.FloatTensor(word_embed))
		for sentence in tqdm(syntactic_embeddings):
			for word_embed in sentence:
				embeddings2.append(torch.FloatTensor(word_embed))

		embeddings = torch.stack(embeddings)
		embeddings2 = torch.stack(embeddings2)

		embeddings = torch.cat([embeddings, embeddings2], dim=0)
		embeddings = F.normalize(embeddings, p=2, dim=1).tolist()

		for embedding in tqdm(embeddings):
			f.write('\t'.join([str(val) for val in embedding])+'\n')
		

		ff.write('Word\tPOS\n')
		for sentence, sentence_tags in tqdm(zip(words, tags)):
			for word, tag in zip(sentence, sentence_tags):
				ff.write('original_' + word + '\t' + 'original_' + tag + '\n')

		for sentence, sentence_tags in tqdm(zip(words, tags)):
			for word, tag in zip(sentence, sentence_tags):
				ff.write('syntactic_' + word + '\t' + 'syntactic_' + tag + '\n')

def write_hdf5(input_path, output_path, model):
	LAYER_COUNT = 12
	FEATURE_COUNT = 768
	BATCH_SIZE = 1

	tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
	# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

	with h5py.File(output_path, 'w') as fout:
		for index, line in enumerate(open(input_path)):
			line = line.strip()
			line = '[CLS] ' + line + ' [SEP]'
			tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
			indexed_tokens = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_text))
			token_start_mask = torch.ByteTensor([1 for x in tokenized_text])
			if torch.cuda.is_available():
				indexed_tokens = indexed_tokens.cuda()
				token_start_mask = token_start_mask.cuda()	
			
			dataset = TextDataset(([indexed_tokens], [token_start_mask], [token_start_mask]))
			loader = DataLoader(dataset=dataset,
								batch_size=BATCH_SIZE)
			embeddings = model.get_embeddings(loader, ignore=False, return_all=True)
			embed = np.array(embeddings[0])

			if index % 1000 == 0:
				print('Processing sentence {}...'.format(index))
			if index < 5:
				print('Len of tokens: {}'.format(len(tokenized_text)))
				print('embed shape: {}\n'.format(embed.shape))
			
			assert len(tokenized_text) == embed.shape[-2]
			
			dset = fout.create_dataset(str(index), (LAYER_COUNT, embed.shape[-2], FEATURE_COUNT))
			dset[:,:,:] = embed


		# This converts all at once but will OOM
		# corpus = Corpus.load(input_path)
		# print('corpus loaded')
		# vocab = Vocab.from_corpus(corpus=corpus, min_freq=2)
		# print('vocab loaded')
		# a, b, c, words, tags = vocab.numericalize_tags(corpus)
		# print('vocab numericalized')
		# dataset = TextDataset((a, b, c))
		# print('dataset loaded')
		# loader = DataLoader(dataset=dataset,
		#                     batch_size=BATCH_SIZE,
		#                     collate_fn=collate_fn)
		# print('loader loaded')
		# embeddings = model.get_embeddings(loader)
		# print('embeddings computed')

		# for index, (sentence, embed) in tqdm(enumerate(zip(words, embeddings))):
		# 	dset = fout.create_dataset(str(index), (LAYER_COUNT, len(sentence), FEATURE_COUNT))
		# 	embed = np.array(embed)
		# 	dset[:,:,:] = embed


example(sentences)

# PennTreebank('data/dev.conllx', 'embeddings.tsv', 'meta.tsv')

# corpus = {
# 	'train_path': 'data/train',
# 	'dev_path': 'data/dev',
# 	'test_path': 'data/test'
# }

# my_embeddings = {
# 	'train_path': 'data/train.bert-layers.hdf5',
# 	'dev_path': 'data/dev.bert-layers.hdf5',
# 	'test_path': 'data/test.bert-layers.hdf5',
# }

# for input_path, output_path in zip(corpus.values(), my_embeddings.values()):
# 	print(input_path)
# 	print(output_path)
# 	write_hdf5(input_path, output_path, model=syntactic_model)
