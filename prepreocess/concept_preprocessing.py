import pandas as pd
from glob import glob
import json
import numpy as np
import re
import torch
from tqdm import tqdm

dataset_path = 'data'
original_path = 'original_data'
train_path = f'{original_path}/concept_train/*'
train_files = glob(train_path)
test_path = f'{original_path}/concept_eval/*'
test_files = glob(test_path)


test_auto_dataset = []
test_auto_size = []

test_input = []
test_output = []
test_input_size = []
test_output_size = []
test_auto_class = []

train_input = []
train_output = []
train_input_size = []
train_output_size = []
train_auto_class = []

max_len = 30
count = 0

pattern = re.compile(r'\\.+')

for concept in tqdm(train_files):
    files = glob(concept+'/*')
    with open(concept, 'rb') as f:
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

        train_input.append(input_arr)
        # test_auto_dataset.append(input_arr)
        train_output.append(output_arr)
        # test_auto_dataset.append(output_arr)

        train_input_size.append(input_size)
        # test_auto_size.append(input_size)
        train_output_size.append(output_size)
        # test_auto_size.append(output_size)

        a = pattern.search(concept)
        data_class = concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-1] if '10' not in concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0] else concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-2]
        train_auto_class.append(data_class)
        train_auto_class.append(data_class)

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

        train_input.append(input_arr)
        # test_auto_dataset.append(input_arr)
        train_output.append(output_arr)
        # test_auto_dataset.append(output_arr)

        train_input_size.append(input_size)
        # test_auto_size.append(input_size)
        train_output_size.append(output_size)
        # test_auto_size.append(output_size)

        a = pattern.search(concept)
        data_class = concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-1] if '10' not in concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0] else concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-2]
        train_auto_class.append(data_class)
        train_auto_class.append(data_class)

for concept in tqdm(test_files):
    with open(concept, 'rb') as f:
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

        test_input.append(input_arr)
        # test_auto_dataset.append(input_arr)
        test_output.append(output_arr)
        # test_auto_dataset.append(output_arr)

        test_input_size.append(input_size)
        # test_auto_size.append(input_size)
        test_output_size.append(output_size)
        # test_auto_size.append(output_size)

        a = pattern.search(concept)
        data_class = concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-1] if '10' not in concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0] else concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-2]
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

        test_input.append(input_arr)
        # test_auto_dataset.append(input_arr)
        test_output.append(output_arr)
        # test_auto_dataset.append(output_arr)

        test_input_size.append(input_size)
        # test_auto_size.append(input_size)
        test_output_size.append(output_size)
        # test_auto_size.append(output_size)

        a = pattern.search(concept)
        data_class = concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-1] if '10' not in concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0] else concept[a.regs[0][0]:a.regs[0][1]].strip('\\').split('.')[0][:-2]
        test_auto_class.append(data_class)
        test_auto_class.append(data_class)


train_auto_json_data = {
    'input': train_input,
    'output': train_output,
    'input_size': train_input_size,
    'output_size': train_output_size,
    'task': train_auto_class,
}


test_auto_json_data = {
    'input': test_input,
    'output': test_output,
    'input_size': test_input_size,
    'output_size': test_output_size,
    'task': test_auto_class,
}

with open(f'{dataset_path}/train_concept.json', 'w') as f:
    json.dump(train_auto_json_data, f)

with open(f'{dataset_path}/test_concept.json', 'w') as f:
    json.dump(test_auto_json_data, f)