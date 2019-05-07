#! bin/bash

# experiments:
# - UD 1.2 on 7 langauges without POS tags
# -- one parser trained for  each language (BERT9-12)
# - UD 2.0 on same 7 languages without POS tags
# -- one parser for each language BUT trained on that language (Done above)
echo "One parser trained for  each language (BERT9-12, UD_v1.2 and UD_v2.0)"
for treebank in UD_v1.2 UD_v2.0; do
	for language in German English Spanish French Italian Portuguese Swedish; do
		#   --do_lower_case ? maybe not first
		echo $treebank $language "S==T"
		python create_datasets.py                                            \
			--ftrain       data/${treebank}/UD_${language}/train.conllx      \
			--fdev         data/${treebank}/UD_${language}/dev.conllx        \
			--ftest        data/${treebank}/UD_${language}/test.conllx       \
			--ftrain_cache data/binary/${treebank}/UD_${language}/trainset   \
			--fdev_cache   data/binary/${treebank}/UD_${language}/devset     \
			--ftest_cache  data/binary/${treebank}/UD_${language}/testset    \
			--vocab        vocabs/${treebank}/${language}.pt                 \
			--bert_model   bert-base-multilingual-cased                      \
		&& CUDA_VISIBLE_DEVICES=0,1,2,3                                      \
		python -m torch.distributed.launch --nproc_per_node=4                \
		    run.py train   								                     \
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
	done
done


# - UD 1.2 on 7 langauges without POS tags
# -- one universal parser for all 7 of them  (BERT9-12)
echo "One universal parser for all 7 of them  (BERT9-12, UD_v1.2)"
treebank=UD_v1.2
language=Universal
echo $treebank $language "Universal Parser"
python create_datasets.py                                            \
	--ftrain       data/${treebank}/UD_${language}/train.conllx      \
	--fdev         data/${treebank}/UD_${language}/dev.conllx        \
	--ftest        data/${treebank}/UD_${language}/test.conllx       \
	--ftrain_cache data/binary/${treebank}/UD_${language}/trainset   \
	--fdev_cache   data/binary/${treebank}/UD_${language}/devset     \
	--ftest_cache  data/binary/${treebank}/UD_${language}/testset    \
	--vocab        vocabs/${language}.pt                             \
	--bert_model   bert-base-multilingual                            \
&& CUDA_VISIBLE_DEVICES=0,1,2,3                                      \
python -m torch.distributed.launch --nproc_per_node=4                \
    run.py train   								                     \
	--checkpoint_dir checkpoints/${treebank}/UD_${language}/         \
	--ftrain         data/${treebank}/UD_${language}/train.conllx    \
	--fdev           data/${treebank}/UD_${language}/dev.conllx      \
	--ftest          data/${treebank}/UD_${language}/test.conllx     \
	--ftrain_cache   data/binary/${treebank}/UD_${language}/trainset \
	--fdev_cache     data/binary/${treebank}/UD_${language}/devset   \
	--ftest_cache    data/binary/${treebank}/UD_${language}/testset  \
	--vocab          vocabs/${language}.pt                           \
	--bert_model     bert-base-multilingual                          \
	--use_lstm     
for language in German English Spanish French Italian Portuguese Swedish; do
	echo $language
    python run.py evaluate   								         \
    --checkpoint_dir checkpoints/${treebank}/UD_${language}   		 \
    --vocab          vocabs/${language}.pt    					     \
    --fdata          data/${treebank}/UD_${language}/test.conllx     \
    --use_lstm 
done


# - UD 2.0 on same 7 languages without POS tags
# -- one parser for each language BUT trained on 6 other languages
treebank=UD_v2.0
for language in German English Spanish French Italian Portuguese Swedish; do
	echo $treebank $language "Reverse"
	python create_datasets.py                                                    \
		--ftrain       data/${treebank}/UD_${language}/reverse/train.conllx      \
		--fdev         data/${treebank}/UD_${language}/reverse/dev.conllx        \
		--ftest        data/${treebank}/UD_${language}/test.conllx               \
		--ftrain_cache data/binary/${treebank}/UD_${language}/reverse/trainset   \
		--fdev_cache   data/binary/${treebank}/UD_${language}/reverse/devset     \
		--ftest_cache  data/binary/${treebank}/UD_${language}/testset            \
		--vocab        vocabs/${language}-reverse.pt                             \
		--bert_model   bert-base-multilingual                                    \
	&& CUDA_VISIBLE_DEVICES=0,1,2,3                                              \
	python -m torch.distributed.launch --nproc_per_node=4                        \
	    run.py train   								                             \
		--checkpoint_dir checkpoints/${treebank}/UD_${language}-reverse/         \
		--ftrain         data/${treebank}/UD_${language}/reverse/train.conllx    \
		--fdev           data/${treebank}/UD_${language}/reverse/dev.conllx      \
		--ftest          data/${treebank}/UD_${language}/test.conllx             \
		--ftrain_cache   data/binary/${treebank}/UD_${language}/reverse/trainset \
		--fdev_cache     data/binary/${treebank}/UD_${language}/reverse/devset   \
		--ftest_cache    data/binary/${treebank}/UD_${language}/testset          \
		--vocab          vocabs/${language}.pt                                   \
		--bert_model     bert-base-multilingual                                  \
		--use_lstm     
done


# - UD 2.0 on same 7 languages with GOLD POS tags
# -- one parser for each language BUT trained on that language
echo "One parser trained for  each language (BERT9-12, UD_v2.0 with POS)"
for treebank in UD_v2.0_POS; do
	for language in German English Spanish French Italian Portuguese Swedish; do
		echo $treebank $language "S==T"
		python create_datasets.py                                            \
			--ftrain       data/${treebank}/UD_${language}/train.conllx      \
			--fdev         data/${treebank}/UD_${language}/dev.conllx        \
			--ftest        data/${treebank}/UD_${language}/test.conllx       \
			--ftrain_cache data/binary/${treebank}/UD_${language}/trainset   \
			--fdev_cache   data/binary/${treebank}/UD_${language}/devset     \
			--ftest_cache  data/binary/${treebank}/UD_${language}/testset    \
			--vocab        vocabs/${language}.pt                             \
			--bert_model   bert-base-multilingual                            \
			--use_pos                                                        \
		&& CUDA_VISIBLE_DEVICES=0,1,2,3                                      \
		python -m torch.distributed.launch --nproc_per_node=4                \
		    run.py train   								                     \
			--checkpoint_dir checkpoints/${treebank}/UD_${language}/         \
			--ftrain         data/${treebank}/UD_${language}/train.conllx    \
			--fdev           data/${treebank}/UD_${language}/dev.conllx      \
			--ftest          data/${treebank}/UD_${language}/test.conllx     \
			--ftrain_cache   data/binary/${treebank}/UD_${language}/trainset \
			--fdev_cache     data/binary/${treebank}/UD_${language}/devset   \
			--ftest_cache    data/binary/${treebank}/UD_${language}/testset  \
			--vocab          vocabs/${language}.pt                           \
			--bert_model     bert-base-multilingual                          \
			--use_pos                                                        \
			--use_lstm     
	done
done

# -- one parser for each language BUT trained on 6 other languages
treebank=UD_v2.0_POS
for language in German English Spanish French Italian Portuguese Swedish; do
	echo $treebank $language "Reverse"
	python create_datasets.py                                                    \
		--ftrain       data/${treebank}/UD_${language}/reverse/train.conllx      \
		--fdev         data/${treebank}/UD_${language}/reverse/dev.conllx        \
		--ftest        data/${treebank}/UD_${language}/test.conllx               \
		--ftrain_cache data/binary/${treebank}/UD_${language}/reverse/trainset   \
		--fdev_cache   data/binary/${treebank}/UD_${language}/reverse/devset     \
		--ftest_cache  data/binary/${treebank}/UD_${language}/testset            \
		--vocab        vocabs/${language}-reverse.pt                             \
		--bert_model   bert-base-multilingual                                    \
		--use_pos                                                                \
	&& CUDA_VISIBLE_DEVICES=0,1,2,3                                              \
	python -m torch.distributed.launch --nproc_per_node=4                        \
	    run.py train   								                             \
		--checkpoint_dir checkpoints/${treebank}/UD_${language}-reverse/         \
		--ftrain         data/${treebank}/UD_${language}/reverse/train.conllx    \
		--fdev           data/${treebank}/UD_${language}/reverse/dev.conllx      \
		--ftest          data/${treebank}/UD_${language}/test.conllx             \
		--ftrain_cache   data/binary/${treebank}/UD_${language}/reverse/trainset \
		--fdev_cache     data/binary/${treebank}/UD_${language}/reverse/devset   \
		--ftest_cache    data/binary/${treebank}/UD_${language}/testset          \
		--vocab          vocabs/${language}.pt                                   \
		--bert_model     bert-base-multilingual                                  \
		--use_pos                                                                \
		--use_lstm     
done
