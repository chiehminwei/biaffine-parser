echo "One parser trained for  each language (BERT9-12, UD_v1.2 with POS)"
for treebank in UD_v1.2; do
	for language in French; do
		echo $treebank $language "S==T"
		CUDA_VISIBLE_DEVICES=0                                               \
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
			--use_pos                                                        \
		    --use_lstm                                                       
	done
done