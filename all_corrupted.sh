#! /bin/bash


# path="debugging";
# for conllx_file in ${path}/*.conllx; do
# 	filename=$(echo $conllx_file | cut -d '/' -f 2 | cut -d '.' -f 1)
# 	echo ' '
# 	echo '***************************************'
# 	echo $filename
# 	echo '***************************************'
# 	echo ' '
# 	CUDA_VISIBLE_DEVICES=2,3 python run.py train --ftrain=$conllx_file --ftrain_cache=all_trainset_debug/$filename --vocab=all_vocab_debug/$filename && echo $filename >> success_debug || echo $filename >> failure_debug
# 	echo '***************************************'
# 	echo ' '
# done

path="all";
for conllx_file in ${path}/*.conllx; do
        filename=$(echo $conllx_file | cut -d '/' -f 2 | cut -d '.' -f 1)
        case $filename in
                UD_Catalan-AnCora|UD_Czech-CLTT|UD_Czech-PDT|UD_French-GSD|UD_French-ParTUT|UD_Italian-ISDT|UD_Italian-ParTUT|UD_Urdu-UDTB)
                        echo ' '
                        echo '***************************************'
                        echo $filename
                        echo '***************************************'
                        echo ' '
                        CUDA_VISIBLE_DEVICES=2,3 python run.py train --ftrain=$conllx_file --ftrain_cache=all_trainset/$filename --vocab=all_vocab/$filename && echo $filename >> success_round2 || echo $filename >> failure_round2
                        echo '***************************************'
                        echo ' ';;
                *) :;;
        esac
done