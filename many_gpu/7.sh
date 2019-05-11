# - UD 1.2 on 7 langauges without POS tags
# -- one universal parser for all 7 of them  (BERT9-12)
echo "One universal parser for all 7 of them  (BERT9-12, UD_v1.2, without POS)"
treebank=UD_v1.2
language=Universal
echo $treebank $language "Universal Parser"
CUDA_VISIBLE_DEVICES=7                                               \
python create_datasets.py                                            \
	--ftrain       data/${treebank}/UD_${language}/train.conllx      \
	--fdev         data/${treebank}/UD_${language}/dev.conllx        \
	--ftest        data/${treebank}/UD_${language}/test.conllx       \
	--ftrain_cache data/binary/${treebank}/UD_${language}/trainset   \
	--fdev_cache   data/binary/${treebank}/UD_${language}/devset     \
	--ftest_cache  data/binary/${treebank}/UD_${language}/testset    \
	--vocab        vocabs/${treebank}/${language}.pt                 \
	--bert_model   bert-base-multilingual-cased                      \
&& CUDA_VISIBLE_DEVICES=7                                            \
python run.py train   								                 \
	--checkpoint_dir checkpoints/${treebank}/UD_${language}/         \
	--ftrain         data/${treebank}/UD_${language}/train.conllx    \
	--fdev           data/${treebank}/UD_${language}/dev.conllx      \
	--ftest          data/${treebank}/UD_${language}/test.conllx     \
	--ftrain_cache   data/binary/${treebank}/UD_${language}/trainset \
	--fdev_cache     data/binary/${treebank}/UD_${language}/devset   \
	--ftest_cache    data/binary/${treebank}/UD_${language}/testset  \
	--vocab          vocabs/${treebank}/${language}.pt               \
	--bert_model     bert-base-multilingual-cased                    \
	--use_lstm     
for language in German English Spanish French Italian Portuguese Swedish; do
	echo $language
    python run.py evaluate   								         \
    --checkpoint_dir checkpoints/${treebank}/UD_${language}   		 \
    --vocab          vocabs/${treebank}/Universal.pt    	         \
    --fdata          data/${treebank}/UD_${language}/test.conllx     \
    --use_lstm 
done



echo "One universal parser for all 7 of them  (BERT9-12, UD_v1.2, with POS)"
treebank=UD_v1.2
language=Universal
echo $treebank $language "Universal Parser"
CUDA_VISIBLE_DEVICES=7                                               \
python create_datasets.py                                            \
	--ftrain       data/${treebank}/UD_${language}/train.conllx      \
	--fdev         data/${treebank}/UD_${language}/dev.conllx        \
	--ftest        data/${treebank}/UD_${language}/test.conllx       \
	--ftrain_cache data/binary/${treebank}/UD_${language}/trainset   \
	--fdev_cache   data/binary/${treebank}/UD_${language}/devset     \
	--ftest_cache  data/binary/${treebank}/UD_${language}/testset    \
	--vocab        vocabs/${treebank}/${language}.pt                 \
	--bert_model   bert-base-multilingual-cased                      \
&& CUDA_VISIBLE_DEVICES=7                                            \
python run.py train   								                 \
	--checkpoint_dir checkpoints/${treebank}/UD_${language}/POS/     \
	--ftrain         data/${treebank}/UD_${language}/train.conllx    \
	--fdev           data/${treebank}/UD_${language}/dev.conllx      \
	--ftest          data/${treebank}/UD_${language}/test.conllx     \
	--ftrain_cache   data/binary/${treebank}/UD_${language}/trainset \
	--fdev_cache     data/binary/${treebank}/UD_${language}/devset   \
	--ftest_cache    data/binary/${treebank}/UD_${language}/testset  \
	--vocab          vocabs/${treebank}/${language}.pt               \
	--bert_model     bert-base-multilingual-cased                    \
	--use_lstm                                                       \
	--use_pos
for language in German English Spanish French Italian Portuguese Swedish; do
	echo $language
    python run.py evaluate   								         \
    --checkpoint_dir checkpoints/${treebank}/UD_${language}/POS/	 \
    --vocab          vocabs/${treebank}/Universal.pt    	         \
    --fdata          data/${treebank}/UD_${language}/test.conllx     \
    --use_lstm                                                       \
    --use_pos
done