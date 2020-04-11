To run:


pip3 install textblob

pip3 install pyconll

pip3 install pandas

pip3 install sklearn

pip3 install openpyxl

python3 run.py


**Use arguments:**


| Argument | Description | Default value  |
| ------------- | :------------- | ----- |
| --context n | Word count before and after target | n = 3 |
| --iter n | Number of training iterations | n = 5 |
| --suffix | Add this argument to use suffix analysis in training | False |
| --part_tag | Add this argument to separate tags into morphological features | False |

*e.g.:*

python3 run.py --context 3 --iter 10 --suffix
