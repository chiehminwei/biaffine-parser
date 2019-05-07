# -*- coding: utf-8 -*-


class Config(object):

    # [Network]
    n_bert_hidden = 768
    n_tag_embed = 100
    bert_dropout = 0.33
    n_lstm_layers = 3
    n_lstm_hidden = 400
    lstm_dropout = 0.33
    n_mlp_arc = 500
    n_mlp_rel = 100
    mlp_dropout = 0.33

    # [Optimizer]
    # lr = 2e-3 # original
    lr = 1e-5 # better for fine-tuning BERT and LM

    # [Run]
    batch_size = 8
    epochs = 1000
    patience = 100
    gradient_accumulation_steps = 1
