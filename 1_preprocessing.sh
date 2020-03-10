#!/bin/bash

pip install pyconll

dev="https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-dev.conllu" 
train="https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-train.conllu"

# To run, type e.g.:   ./1_preprocessing.sh 1 
echo "Processing file" $dev
python ./1_preprocessing.py --file $dev --context $1

echo "Processing file" $train
python ./1_preprocessing.py --file $train --context $1