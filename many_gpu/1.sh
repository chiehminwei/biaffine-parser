echo "- UD 2.0 on same 7 languages with POS tags"
echo "--one parser for each language BUT trained on 6 other languages"
# -- one parser for each language BUT trained on 6 other languages
treebank=UD_v2.0_POS
for language in English; do
	echo $treebank $language "Reverse"
	CUDA_VISIBLE_DEVICES=1                                                       \
	python create_datasets.py                                                    \
		--ftrain       data/${treebank}/UD_${language}/reverse/train.conllx      \
		--fdev         data/${treebank}/UD_${language}/reverse/dev.conllx        \
		--ftest        data/${treebank}/UD_${language}/test.conllx               \
		--ftrain_cache data/binary/${treebank}/UD_${language}/reverse/trainset   \
		--fdev_cache   data/binary/${treebank}/UD_${language}/reverse/devset     \
		--ftest_cache  data/binary/${treebank}/UD_${language}/testset            \
		--vocab        vocabs/${treebank}/${language}-reverse.pt                 \
		--bert_model   bert-base-multilingual-cased                              \
	&& CUDA_VISIBLE_DEVICES=1                                                    \
	python run.py train   								                         \
		--checkpoint_dir checkpoints/${treebank}/UD_${language}-reverse/         \
		--ftrain         data/${treebank}/UD_${language}/reverse/train.conllx    \
		--fdev           data/${treebank}/UD_${language}/reverse/dev.conllx      \
		--ftest          data/${treebank}/UD_${language}/test.conllx             \
		--ftrain_cache   data/binary/${treebank}/UD_${language}/reverse/trainset \
		--fdev_cache     data/binary/${treebank}/UD_${language}/reverse/devset   \
		--ftest_cache    data/binary/${treebank}/UD_${language}/testset          \
		--vocab          vocabs/${treebank}/${language}-reverse.pt               \
		--bert_model     bert-base-multilingual-cased                            \
		--use_pos                                                                \
		--use_lstm     
done


echo "- UD 2.0 on same 7 languages without POS tags"
echo "--one parser for each language BUT trained on 6 other languages"
treebank=UD_v2.0
for language in English; do
	echo $treebank $language "Reverse"
	CUDA_VISIBLE_DEVICES=1                                                       \
	python create_datasets.py                                                    \
		--ftrain       data/${treebank}/UD_${language}/reverse/train.conllx      \
		--fdev         data/${treebank}/UD_${language}/reverse/dev.conllx        \
		--ftest        data/${treebank}/UD_${language}/test.conllx               \
		--ftrain_cache data/binary/${treebank}/UD_${language}/reverse/trainset   \
		--fdev_cache   data/binary/${treebank}/UD_${language}/reverse/devset     \
		--ftest_cache  data/binary/${treebank}/UD_${language}/testset            \
		--vocab        vocabs/${treebank}/${language}-reverse.pt                 \
		--bert_model   bert-base-multilingual-cased                              \
	&& CUDA_VISIBLE_DEVICES=1                                                    \
	python run.py train   								                         \
		--checkpoint_dir checkpoints/${treebank}/UD_${language}-reverse/         \
		--ftrain         data/${treebank}/UD_${language}/reverse/train.conllx    \
		--fdev           data/${treebank}/UD_${language}/reverse/dev.conllx      \
		--ftest          data/${treebank}/UD_${language}/test.conllx             \
		--ftrain_cache   data/binary/${treebank}/UD_${language}/reverse/trainset \
		--fdev_cache     data/binary/${treebank}/UD_${language}/reverse/devset   \
		--ftest_cache    data/binary/${treebank}/UD_${language}/testset          \
		--vocab          vocabs/${treebank}/${language}-reverse.pt               \
		--bert_model     bert-base-multilingual-cased                            \
		--use_lstm     
done


echo "One parser trained for  each language (BERT9-12, UD_v1.2 with POS)"
for treebank in UD_v1.2; do
	for language in English; do
		echo $treebank $language "S==T"
		CUDA_VISIBLE_DEVICES=1                                               \
		python create_datasets.py                                            \
			--ftrain       data/${treebank}/UD_${language}/train.conllx      \
			--fdev         data/${treebank}/UD_${language}/dev.conllx        \
			--ftest        data/${treebank}/UD_${language}/test.conllx       \
			--ftrain_cache data/binary/${treebank}/UD_${language}/trainset   \
			--fdev_cache   data/binary/${treebank}/UD_${language}/devset     \
			--ftest_cache  data/binary/${treebank}/UD_${language}/testset    \
			--vocab        vocabs/${treebank}/${language}.pt                 \
			--bert_model   bert-base-multilingual-cased                      \
		&& CUDA_VISIBLE_DEVICES=1                                            \
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
			--use_lstm                                                       \
			--use_pos
	done
done