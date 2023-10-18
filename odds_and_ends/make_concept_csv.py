import pandas as pd
from glob import glob
import os

dataset_path = '../data'
original_path = '../original_data'
train_path = f'original_data/training/*'
train_files = glob(train_path)

file_name_list = []


for file in train_files:
    file_name = file.split('\\')[-1].split('.')[0]
    file_name_list.append(file_name)

a = pd.DataFrame(file_name_list)
pass