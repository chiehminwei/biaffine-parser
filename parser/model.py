# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser.metric import AttachmentMethod
from parser.parser import BiaffineParser

import torch
import torch.nn as nn
import torch.optim as optim


class Model(object):

    def __init__(self, vocab, network):
        super(Model, self).__init__()

        self.vocab = vocab
        self.network = network
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, loaders, epochs, patience,
                 lr, betas, epsilon, annealing, file):
        total_time = timedelta()
        max_e, max_metric = 0, 0.0
        train_loader, dev_loader, test_loader = loaders
        self.optimizer = optim.Adam(params=self.network.parameters(),
                                    lr=lr, betas=betas, eps=epsilon)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                     lr_lambda=annealing)

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            self.train(train_loader)

            print(f"Epoch {epoch} / {epochs}:")
            loss, train_metric = self.evaluate(train_loader, False)
            print(f"{'train:':<6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self.evaluate(dev_loader, False)
            print(f"{'dev:':<6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self.evaluate(test_loader, False)
            print(f"{'test:':<6} Loss: {loss:.4f} {test_metric}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # save the model if it is the best so far
            if dev_metric > max_metric:
                self.network.save(file)
                max_e, max_metric = epoch, dev_metric
            elif epoch - max_e >= patience:
                break
        self.network = BiaffineParser.load(file)
        loss, metric = self.evaluate(test_loader)

        print(f"max score of dev is {max_metric.score:.2%} at epoch {max_e}")
        print(f"the score of test at epoch {max_e} is {metric.score:.2%}")
        print(f"mean time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def train(self, loader):
        self.network.train()

        for words, tags, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.network(words, tags)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader, include_punct=True):
        self.network.eval()

        loss, metric = 0, AttachmentMethod()

        for words, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # ignore all punctuation if specified
            if not include_punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)
            s_arc, s_rel = self.network(words, tags)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            loss += self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.network.eval()

        all_arcs, all_rels = [], []
        for words, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_arc, s_rel = self.network(words, tags)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

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
