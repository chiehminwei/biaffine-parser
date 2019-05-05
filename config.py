# -*- coding: utf-8 -*-


class Config(object):

    # [Network]
    n_bert_hidden = 768
    bert_dropout = 0.33
    n_mlp_arc = 500
    n_mlp_rel = 100
    mlp_dropout = 0.33

    # [Optimizer]
    lr = 1e-5

    # [Run]
    batch_size = 8
    epochs = 60
    patience = 15
    gradient_accumulation_steps = 1
