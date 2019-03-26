# Biaffine Parser

[![Travis](https://img.shields.io/travis/zysite/biaffine-parser.svg)](https://travis-ci.org/zysite/biaffine-parser)
[![LICENSE](https://img.shields.io/github/license/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/blob/master/LICENSE)	
[![GitHub issues](https://img.shields.io/github/issues/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/issues)		
[![GitHub stars](https://img.shields.io/github/stars/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/stargazers)

An implementation of "Deep Biaffine Attention for Neural Dependency Parsing".
Use the char branch for BERT.
Code based on https://github.com/zysite/biaffine-parser



Details and [hyperparameter choices](#Hyperparameters) are almost identical to those described in the paper, except for some training settings. Also, we do not provide a decoding algorithm to ensure well-formedness, and this does not seriously affect the results.

Another version of the implementation is available on [char](https://github.com/zysite/biaffine-parser/tree/char) branch, which replaces the tag embedding with char lstm and achieves better performance.

## Requirements

```txt
python == 3.7.0
pytorch == 1.0.0
```

## Datasets

|                   |   train |   dev |   test |
|-------------------|---------|-------|--------|
| UD_English-ESL    |    4124 |   500 |    500 |
| UD_English-EWT    |   12543 |  2002 |   2077 |
| UD_English-GUM    |    2914 |   707 |    778 |
| UD_English-LinES  |    2738 |   912 |    914 |
| UD_English-PUD    |         |       |   1000 |
| UD_English-ParTUT |    1781 |   156 |    153 |


|                       Corpus  | # Tree|  
|-------------------------------|-------|
| UD_Afrikaans-AfriBooms        |  1934 |
| UD_Akkadian-PISANDUB          |   101 |
| UD_Amharic-ATT                |  1074 |
| UD_Ancient_Greek-PROIEL       | 17081 |
| UD_Ancient_Greek-Perseus      | 13919 |
| UD_Arabic-NYUAD               | 19738 |
| UD_Arabic-PADT                |  7664 |
| UD_Arabic-PUD                 |  1000 |
| UD_Armenian-ArmTDP            |  1030 |
| UD_Bambara-CRB                |  1026 |
| UD_Basque-BDT                 |  8993 |
| UD_Belarusian-HSE             |   393 |
| UD_Breton-KEB                 |   888 |
| UD_Bulgarian-BTB              | 11138 |
| UD_Buryat-BDT                 |   927 |
| UD_Cantonese-HK               |   650 |
| UD_Catalan-AnCora             | 16678 |
| UD_Chinese-CFL                |   451 |
| UD_Chinese-GSD                |  4997 |
| UD_Chinese-HK                 |   908 |
| UD_Chinese-PUD                |  1000 |
| UD_Coptic-Scriptorium         |   840 |
| UD_Croatian-SET               |  8889 |
| UD_Czech-CAC                  | 24709 |
| UD_Czech-CLTT                 |  1125 |
| UD_Czech-FicTree              | 12760 |
| UD_Czech-PDT                  | 87913 |
| UD_Czech-PUD                  |  1000 |
| UD_Danish-DDT                 |  5512 |
| UD_Dutch-Alpino               | 13583 |
| UD_Dutch-LassySmall           |  7341 |
| UD_English-ESL                |  5124 |
| UD_English-EWT                | 16622 |
| UD_English-GUM                |  4399 |
| UD_English-LinES              |  4564 |
| UD_English-PUD                |  1000 |
| UD_English-ParTUT             |  2090 |
| UD_Erzya-JR                   |  1550 |
| UD_Estonian-EDT               | 30723 |
| UD_Faroese-OFT                |  1208 |
| UD_Finnish-FTB                | 18723 |
| UD_Finnish-PUD                |  1000 |
| UD_Finnish-TDT                | 15136 |
| UD_French-FTB                 | 18535 |
| UD_French-GSD                 | 16342 |
| UD_French-PUD                 |  1000 |
| UD_French-ParTUT              |  1020 |
| UD_French-Sequoia             |  3099 |
| UD_French-Spoken              |  2786 |
| UD_Galician-CTG               |  3993 |
| UD_Galician-TreeGal           |  1000 |
| UD_German-GSD                 | 15590 |
| UD_German-PUD                 |  1000 |
| UD_Gothic-PROIEL              |  5401 |
| UD_Greek-GDT                  |  2521 |
| UD_Hebrew-HTB                 |  6216 |
| UD_Hindi-HDTB                 | 16647 |
| UD_Hindi-PUD                  |  1000 |
| UD_Hindi_English-HIENCS       |  1898 |
| UD_Hungarian-Szeged           |  1800 |
| UD_Indonesian-GSD             |  5593 |
| UD_Indonesian-PUD             |  1000 |
| UD_Irish-IDT                  |  1020 |
| UD_Italian-ISDT               | 14167 |
| UD_Italian-PUD                |  1000 |
| UD_Italian-ParTUT             |  2090 |
| UD_Italian-PoSTWITA           |  6713 |
| UD_Japanese-BCCWJ             | 57109 |
| UD_Japanese-GSD               |  8195 |
| UD_Japanese-Modern            |   822 |
| UD_Japanese-PUD               |  1000 |
| UD_Kazakh-KTB                 |  1078 |
| UD_Komi_Zyrian-IKDP           |    87 |
| UD_Komi_Zyrian-Lattice        |   190 |
| UD_Korean-GSD                 |  6339 |
| UD_Korean-Kaist               | 27363 |
| UD_Korean-PUD                 |  1000 |
| UD_Kurmanji-MG                |   754 |
| UD_Latin-ITTB                 | 21011 |
| UD_Latin-PROIEL               | 18400 |
| UD_Latin-Perseus              |  2273 |
| UD_Latvian-LVTB               |  9920 |
| UD_Lithuanian-HSE             |   263 |
| UD_Maltese-MUDT               |  2074 |
| UD_Marathi-UFAL               |   466 |
| UD_Naija-NSC                  |   948 |
| UD_North_Sami-Giella          |  3122 |
| UD_Norwegian-Bokmaal          | 20045 |
| UD_Norwegian-Nynorsk          | 17575 |
| UD_Norwegian-NynorskLIA       |  1396 |
| UD_Old_Church_Slavonic-PROIEL |  6337 |
| UD_Old_French-SRCMF           | 17678 |
| UD_Persian-Seraji             |  5997 |
| UD_Polish-LFG                 | 17246 |
| UD_Polish-SZ                  |  8227 |
| UD_Portuguese-Bosque          |  9365 |
| UD_Portuguese-GSD             | 12078 |
| UD_Portuguese-PUD             |  1000 |
| UD_Romanian-Nonstandard       | 10069 |
| UD_Romanian-RRT               |  9524 |
| UD_Russian-GSD                |  5030 |
| UD_Russian-PUD                |  1000 |
| UD_Russian-SynTagRus          | 61889 |
| UD_Russian-Taiga              |  1764 |
| UD_Sanskrit-UFAL              |   230 |
| UD_Serbian-SET                |  3891 |
| UD_Slovak-SNK                 | 10604 |
| UD_Slovenian-SSJ              |  8000 |
| UD_Slovenian-SST              |  3188 |
| UD_Spanish-AnCora             | 17680 |
| UD_Spanish-GSD                | 16013 |
| UD_Spanish-PUD                |  1000 |
| UD_Swedish-LinES              |  4564 |
| UD_Swedish-PUD                |  1000 |
| UD_Swedish-Talbanken          |  6026 |
| UD_Swedish_Sign_Language-SSLC |   203 |
| UD_Tagalog-TRG                |    55 |
| UD_Tamil-TTB                  |   600 |
| UD_Telugu-MTG                 |  1328 |
| UD_Thai-PUD                   |  1000 |
| UD_Turkish-IMST               |  5635 |
| UD_Turkish-PUD                |  1000 |
| UD_Ukrainian-IU               |  6801 |
| UD_Upper_Sorbian-UFAL         |   646 |
| UD_Urdu-UDTB                  |  5130 |
| UD_Uyghur-UDT                 |  3456 |
| UD_Vietnamese-VTB             |  3000 |
| UD_Warlpiri-UFAL              |    55 |
| UD_Yoruba-YTB                 |   100 |


The model is evaluated on the Stanford Dependency conversion ([v3.3.0](https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip)) of the English Penn Treebank with POS tags predicted by [Stanford POS tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip).

For all datasets, we follow the conventional data splits:

* Train: 02-21 (39,832 sentences)
* Dev: 22 (1,700 sentences)
* Test: 23 (2,416 sentences)

## Performance

|               |  UAS  |  LAS  |
| ------------- | :---: | :---: |
| tag embedding | 95.87 | 94.19 |
| char lstm     | 96.17 | 94.53 |

Note that punctuation is excluded in all evaluation metrics. 

Aside from using consistent hyperparameters, there are some keypoints that significantly affect the performance:

- Dividing the pretrained embedding by its standard-deviation
- Applying the same dropout mask at every recurrent timestep
- Jointly dropping the words and tags

For the above reasons, we may have to give up some native modules in pytorch (e.g., `LSTM` and `Dropout`), and use self-implemented ones instead.

As shown above, our results, especially on char lstm version, have outperformed the [offical implementation](https://github.com/tdozat/Parser-v1) (95.74 and 94.08).

## Usage

You can start the training, evaluation and prediction process by using subcommands registered in `parser.commands`.

```sh
$ python run.py -h
usage: run.py [-h] {evaluate,predict,train} ...

Create the Biaffine Parser model.

optional arguments:
  -h, --help            show this help message and exit

Commands:
  {evaluate,predict,train}
    evaluate            Evaluate the specified model and dataset.
    predict             Use a trained model to make predictions.
    train               Train a model.
```

Before triggering the subparser, please make sure that the data files must be in CoNLL-X format. If some fields are missing, you can use underscores as placeholders.

Optional arguments of the subparsers are as follows:

```sh
$ python run.py train -h
usage: run.py train [-h] [--ftrain FTRAIN] [--fdev FDEV] [--ftest FTEST]
                    [--fembed FEMBED] [--device DEVICE] [--seed SEED]
                    [--threads THREADS] [--file FILE] [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --ftrain FTRAIN       path to train file
  --fdev FDEV           path to dev file
  --ftest FTEST         path to test file
  --fembed FEMBED       path to pretrained embedding file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file

$ python run.py evaluate -h
usage: run.py evaluate [-h] [--batch-size BATCH_SIZE] [--include-punct]
                       [--fdata FDATA] [--device DEVICE] [--seed SEED]
                       [--threads THREADS] [--file FILE] [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --include-punct       whether to include punctuation
  --fdata FDATA         path to dataset
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file

$ python run.py predict -h
usage: run.py predict [-h] [--batch-size BATCH_SIZE] [--fdata FDATA]
                      [--fpred FPRED] [--device DEVICE] [--seed SEED]
                      [--threads THREADS] [--file FILE] [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --fdata FDATA         path to dataset
  --fpred FPRED         path to predicted result
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file
```

## Hyperparameters

| Param         | Description                             |                                 Value                                  |
| :------------ | :-------------------------------------- | :--------------------------------------------------------------------: |
| n_embed       | dimension of word embedding             |                                  100                                   |
| n_tag_embed   | dimension of tag embedding              |                                  100                                   |
| embed_dropout | dropout ratio of embeddings             |                                  0.33                                  |
| n_lstm_hidden | dimension of lstm hidden state          |                                  400                                   |
| n_lstm_layers | number of lstm layers                   |                                   3                                    |
| lstm_dropout  | dropout ratio of lstm                   |                                  0.33                                  |
| n_mlp_arc     | arc mlp size                            |                                  500                                   |
| n_mlp_rel     | label mlp size                          |                                  100                                   |
| mlp_dropout   | dropout ratio of mlp                    |                                  0.33                                  |
| lr            | starting learning rate of training      |                                  2e-3                                  |
| betas         | hyperparameter of momentum and L2 norm  |                               (0.9, 0.9)                               |
| epsilon       | stability constant                      |                                 1e-12                                  |
| annealing     | formula of learning rate annealing      | <img src="https://latex.codecogs.com/gif.latex?.75^{\frac{t}{5000}}"/> |
| batch_size    | number of sentences per training update |                                  200                                   |
| epochs        | max number of epochs                    |                                  1000                                  |
| patience      | patience for early stop                 |                                  100                                   |

## References

* [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
 