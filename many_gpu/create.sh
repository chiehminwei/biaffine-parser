#! bin/bash

echo "- UD 2.0 on same 7 languages without POS tags"
echo "--one parser for each language BUT trained on 6 other languages"
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
		--vocab        vocabs/${treebank}/${language}-reverse.pt                 \
		--bert_model   bert-base-multilingual-cased                              
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
	--vocab        vocabs/${treebank}/${language}.pt                 \
	--bert_model   bert-base-multilingual-cased                          

# # experiments:
# # - UD 1.2 on 7 langauges without POS tags
# # -- one parser trained for  each language (BERT9-12)
# # - UD 2.0 on same 7 languages without POS tags
# # -- one parser for each language BUT trained on that language (Done above)
# echo "One parser trained for  each language (BERT9-12, UD_v1.2 and UD_v2.0)"
for treebank in UD_v1.2 UD_v2.0; do
	for language in German English Spanish French Italian Portuguese Swedish; do
		echo $treebank $language "S==T"
		python create_datasets.py                                            \
			--ftrain       data/${treebank}/UD_${language}/train.conllx      \
			--fdev         data/${treebank}/UD_${language}/dev.conllx        \
			--ftest        data/${treebank}/UD_${language}/test.conllx       \
			--ftrain_cache data/binary/${treebank}/UD_${language}/trainset   \
			--fdev_cache   data/binary/${treebank}/UD_${language}/devset     \
			--ftest_cache  data/binary/${treebank}/UD_${language}/testset    \
			--vocab        vocabs/${treebank}/${language}.pt                 \
			--bert_model   bert-base-multilingual-cased                      
	done
done