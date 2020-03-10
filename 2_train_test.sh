#!/bin/bash

pip install textblob

# To run training, type e.g.:   ./2_train_test.sh lv_lvtb-ud-train_result.tsv true
# To run a test, type e.g.:   ./2_train_test.sh lv_lvtb-ud-dev_result.tsv false
echo "Starting"
python ./2_train_test.py --data $1 --train $2
echo "Finished"