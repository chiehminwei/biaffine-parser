# -*- coding: utf-8 -*-

import argparse
import regex

from collections import namedtuple
from collections import defaultdict

'''
Calculate recall for some clases of dependency relations

We consider the recall of 
    left attachments (where the head word precedes the dependent word in the sentence),
    right attachments, 
    root attachments, 
    short-attachments (with distance = 1), 
    long attachments (with distance>6), 
    as well as the following relation groups: 
        nsubj (nominal subjects:nsubj,nsubjpass), 
        dobj (direct object:dobj),
        conj (conjunct:conj), 
        comp (clausal complements:ccomp,xcomp), 
        case (clitics and adpositions:case), 
        mod(modifiers of a noun:nmod,nummod,amod,appos)
'''

Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL', 'LANGUAGE'])


class Corpus(object):
    ROOT = '<ROOT>'

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i)) for i in zip(*sentence)) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        return [[self.ROOT] + [word for word in sentence.FORM]
                for sentence in self.sentences]

    @property
    def heads(self):
        return [[0] + list(map(int, sentence.HEAD))
                for sentence in self.sentences]
 
    @property
    def tags(self):
        return [[self.ROOT] + list(sentence.CPOS)
                for sentence in self.sentences]
                
    @property
    def rels(self):
        return [[self.ROOT] + list(sentence.DEPREL)
                for sentence in self.sentences]

    @property
    def langs(self):
        return [[self.ROOT] + list(sentence.LANGUAGE)
                for sentence in self.sentences]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(HEAD=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(DEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname):
        start, sentences = 0, []
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line[0] == '#':
                start += 1
            if len(line) <= 1:
                try:
                    sentence = Sentence(*zip(*[l.split('\t') for l in lines[start:i] if "." not in l.split('\t')[0] and "-" not in l.split('\t')[0]]))
                    if len(sentence.ID) > 0:
                        sentences.append(sentence)
                except:
                    pass
                start = i + 1
        assert len(sentences) > 0
        corpus = cls(sentences)

        return corpus


parser = argparse.ArgumentParser(
    description='Data preprocessing module.'
)

parser.add_argument('--gold', default='data/dev.conllx',
                    help='path to train file')
parser.add_argument('--test', default='data/dev.conllx',
                    help='path to prediction file')
args = parser.parse_args() 

gold_corpus = Corpus.load(args.gold)
test_corpus = Corpus.load(args.test)


tleft, left = 0, 0
tright, right = 0, 0
troot, root = 0, 0
tshort, short = 0, 0
t_long, _long = 0, 0
tnsubj, nsubj = 0, 0
tdobj, dobj = 0, 0
tconj, conj = 0, 0
tcomp, comp = 0, 0
tcase, case = 0, 0
tmod, mod = 0, 0
print(len(gold_corpus.sentences), len(test_corpus.sentences))
assert len(gold_corpus.sentences) == len(test_corpus.sentences)
for gold_sentence, test_sentence in zip(gold_corpus.sentences, test_corpus.sentences):
    # TODO: ignore all punctuation
    words = gold_sentence.FORM
    
    gold_rels = gold_sentence.DEPREL
    test_rels = test_sentence.DEPREL
    assert len(gold_rels) == len(test_rels)
    for word, gold_rel, test_rel in zip(words, gold_rels, test_rels):
        # Ignore punctuation
        if regex.match(r'\p{P}+$', word):
            continue
        if gold_rel == 'nsubj' or gold_rel == 'nsubjpass':
            nsubj += 1
            # if test_rel == 'nsubj' or test_rel == 'nsubjpass':
            if test_rel == gold_rel:
                tnsubj += 1
        elif gold_rel == 'dobj':
            dobj += 1
            if test_rel == 'dobj':
                tdobj += 1
        elif gold_rel == 'conj':
            conj += 1
            if test_rel == 'conj':
                tconj += 1
        elif gold_rel == 'ccomp' or gold_rel == 'xcomp':
            comp += 1
            if test_rel == gold_rel:
            # if test_rel == 'ccomp' or test_rel == 'xcomp':
                tcomp += 1
        elif gold_rel == 'case':
            case += 1
            if test_rel == 'case':
                tcase += 1
        elif gold_rel == 'nmod' or gold_rel =='nummod' or gold_rel == 'amod' or gold_rel == 'appos':
            mod += 1
            if test_rel == gold_rel:
            # if test_rel == 'nmod' or test_rel =='nummod' or test_rel == 'amod' or test_rel == 'appos':
                tmod += 1

    gold_heads = gold_sentence.HEAD
    test_heads = test_sentence.HEAD

    position = 0
    for gold_head, test_head in zip(gold_heads, test_heads):
        position += 1
        gold_head, test_head = int(gold_head), int(test_head)
        if gold_head > position:
            left += 1
            if gold_head == test_head:
                tleft += 1

        elif gold_head < position:
            right += 1
            if gold_head == test_head:
                tright += 1
        if gold_head == 0:
            root += 1
            if gold_head == test_head:
                troot += 1

        abs_distance = abs(gold_head - position)
        if abs_distance == 1:
            short += 1
            if gold_head == test_head:
                tshort += 1
        if abs_distance > 6:
            _long += 1
            if gold_head == test_head:
                t_long += 1

print('left {0:.2f}'.format(tleft/left*100))
print('right {0:.2f}'.format(tright/right*100))
print('root {0:.2f}'.format(troot/root*100))
print('short {0:.2f}'.format(tshort/short*100))
print('long {0:.2f}'.format(t_long/_long*100))
print('nsubj {0:.2f}'.format(tnsubj/nsubj*100))
print('dobj {0:.2f}'.format(tdobj/dobj*100))
print('conj {0:.2f}'.format(tconj/conj*100))
print('comp {0:.2f}'.format(tcomp/comp*100))
print('case {0:.2f}'.format(tcase/case*100))
print('mod {0:.2f}'.format(tmod/mod*100))
