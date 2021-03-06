To run:


pip3 install textblob pyconll pandas sklearn openpyxl RandomWords

python3 run.py


**Use arguments:**


| Argument | Description | Default value  |
| ------------- | :------------- | ----- |
| --context n | Word count before and after target | n = 3 |
| --iter n | Number of training iterations | n = 5 |
| --suffix | Add this argument to use suffix analysis in training | False |
| --random | Add this argument to generate random english wordsinstead of all target words | False |
| --part_tag | Add this argument to separate tags into morphological features | False |
| --tag_iter | Add this argument to tag the test data set after each training iteration | False |
| --conf_m | Add this argument to create a confusion matrix | False |
| --use_dropout | Add this argument to use randomised feature dropout | False |
| --use_comb | Add this argument to use combined features: 1 - one combined feature representing context_len words before and after target; 2 - all combinations | 0 |

*e.g.:*

python3 run.py --context 3 --iter 10 --suffix
