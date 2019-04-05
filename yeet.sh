#! /bin/bash


path="all/";
for conllx_file in ${path}UD_*/*.conllx; do
	echo $conllx_file
    filename=$(echo $conllx_file | cut -d '/' -f 1)
	echo $filename
 #    case $filename in
 #        UD_Afrikaans-AfriBooms|UD_Arabic-NYUAD|UD_Arabic-PADT|UD_Armenian-ArmTDP|UD_Basque-BDT|UD_Belarusian-HSE|UD_Bulgarian-BTB|UD_Catalan-AnCora|UD_Chinese-GSD|UD_Croatian-SET|UD_Czech-CAC|UD_Czech-CLTT|UD_Czech-FicTree|UD_Czech-PDT|UD_Danish-DDT|UD_Dutch-Alpino|UD_Dutch-LassySmall|UD_English-ESL|UD_English-EWT|UD_English-GUM|UD_English-LinES|UD_English-ParTUT|UD_Estonian-EDT|UD_Finnish-FTB|UD_Finnish-TDT|UD_French-GSD|UD_French-ParTUT|UD_French-Sequoia|UD_Galician-CTG|UD_Galician-TreeGal|UD_German-GSD|UD_Greek-GDT|UD_Hebrew-HTB|UD_Hindi-HDTB|UD_Hungarian-Szeged|UD_Indonesian-GSD|UD_Irish-IDT|UD_Italian-ISDT|UD_Italian-ParTUT|UD_Japanese-GSD|UD_Kazakh-KTB|UD_Korean-GSD|UD_Korean-Kaist|UD_Latin-ITTB|UD_Latin-PROIEL|UD_Latin-Perseus|UD_Latvian-LVTB|UD_Lithuanian-HSE|UD_Marathi-UFAL|UD_Norwegian-Bokmaal|UD_Norwegian-Nynorsk|UD_Persian-Seraji|UD_Polish-LFG|UD_Polish-SZ|UD_Portuguese-Bosque|UD_Portuguese-GSD|UD_Romanian-RRT|UD_Russian-GSD|UD_Russian-SynTagRus|UD_Serbian-SET|UD_Slovak-SNK|UD_Slovenian-SSJ|UD_Spanish-AnCora|UD_Spanish-GSD|UD_Swedish-LinES|UD_Swedish-Talbanken|UD_Tamil-TTB|UD_Telugu-MTG|UD_Turkish-IMST|UD_Urdu-UDTB|UD_Vietnamese-VTB)
 #            cat $conllu_file >> ${filename}.conllx;;
 #        *)             echo 'Skipping '$filename;;
 #    esac
done
