import pandas as pd
from glob import glob
import json
import numpy as np
import re
import torch

dataset_path = '../data'
original_path = '../original_data'
test_path = f'{original_path}/corpus/*'
test_files = glob(test_path)


test_auto_dataset = []
test_auto_size = []
test_auto_class = []

test_input = []
test_output = []
test_input_size = []
test_output_size = []

max_len = 30
count = 0

pattern = re.compile(r'\\.+\\')

for concept in test_files:
    files = glob(concept+'/*')
    for file in files:
        if 'json' not in file:
            continue
        with open(file, 'rb') as f:
            data = json.load(f)
        train_data = data['train']
        valid_data = data['test']

        for i in range(len(train_data)):
            train_data[i]['input'] = (np.array(train_data[i]['input'])).tolist()
            train_input_y_len, train_input_x_len = np.array(train_data[i]['input']).shape
            train_output_y_len, train_output_x_len = np.array(train_data[i]['output']).shape
            input_size = (train_input_y_len, train_input_x_len)
            output_size = (train_output_y_len, train_output_x_len)

            input_arr = [
                [0 if m > train_input_x_len - 1 or n > train_input_y_len - 1 else train_data[i]['input'][n][m] + 1 for m
                 in range(max_len)] for n in range(max_len)]
            output_arr = [
                [0 if m > train_output_x_len - 1 or n > train_output_y_len - 1 else train_data[i]['output'][n][m] + 1
                 for m in range(max_len)] for n in range(max_len)]

            # train_input.append(input_arr)
            test_auto_dataset.append(input_arr)
            # train_output.append(output_arr)
            test_auto_dataset.append(output_arr)

            # train_input_size.append(input_size)
            test_auto_size.append(input_size)
            # train_output_size.append(output_size)
            test_auto_size.append(output_size)

            a = pattern.search(file)
            data_class = file[a.regs[0][0]:a.regs[0][1]].strip('\\')
            test_auto_class.append(data_class)
            test_auto_class.append(data_class)

        for i in range(len(valid_data)):
            valid_data[i]['input'] = (np.array(valid_data[i]['input'])).tolist()
            valid_input_y_len, valid_input_x_len = np.array(valid_data[i]['input']).shape
            valid_output_y_len, valid_output_x_len = np.array(valid_data[i]['output']).shape

            input_size = (valid_input_y_len, valid_input_x_len)
            output_size = (valid_output_y_len, valid_output_x_len)

            input_arr = [
                [0 if m > valid_input_x_len - 1 or n > valid_input_y_len - 1 else valid_data[i]['input'][n][m] + 1 for m
                 in range(max_len)] for n in range(max_len)]
            output_arr = [
                [0 if m > valid_output_x_len - 1 or n > valid_output_y_len - 1 else valid_data[i]['output'][n][m] + 1
                 for m in range(max_len)] for n in range(max_len)]

            # valid_input.append(input_arr)
            test_auto_dataset.append(input_arr)
            # valid_output.append(output_arr)
            test_auto_dataset.append(output_arr)

            # valid_input_size.append(input_size)
            test_auto_size.append(input_size)
            # valid_output_size.append(output_size)
            test_auto_size.append(output_size)

            a = pattern.search(file)
            data_class = file[a.regs[0][0]:a.regs[0][1]].strip('\\')
            test_auto_class.append(data_class)
            test_auto_class.append(data_class)


test_auto_json_data = {
    'data': test_auto_dataset,
    'size': test_auto_size,
    'class': test_auto_class
}

with open(f'{dataset_path}/test_concept_auto_data.json', 'w') as f:
    json.dump(test_auto_json_data, f)