import json
import glob
from collections import OrderedDict
import torch

original_dataset = './original_dataset'
evaluation_path = f'{original_dataset}/training/*'
file_list = glob.glob(evaluation_path)
total_len = len(file_list)
correct = 0

for file in file_list:
    temp_dict = {}
    with open(file, 'r') as f:
        data = json.load(f)
    for i in range(len(data['train'])):
        data_shape = torch.tensor(data['train'][i]['output']).shape
        if temp_dict.get(f'{data_shape[0]}, {data_shape[1]}') == None:
            temp_dict[f'{data_shape[0]}, {data_shape[1]}'] = 1
        else:
            temp_dict[f'{data_shape[0]}, {data_shape[1]}'] += 1
    ordered_dict = OrderedDict(sorted(temp_dict.items(), reverse=True))
    predict = list(ordered_dict.keys())[0]

    test_shape = torch.tensor(data['test'][0]['output']).shape
    test_input_size = torch.tensor(data['test'][0]['input']).shape
    test_output_size = f'{test_shape[0]}, {test_shape[1]}'

    if len(list(ordered_dict.keys())) >= 2:
        predict = f'{test_input_size[0]}, {test_input_size[1]}'

    if predict == test_output_size:
        correct += 1
    else:
        pass


print(correct/total_len*100)
print(correct)
print(total_len)