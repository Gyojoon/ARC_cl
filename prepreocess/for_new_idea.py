from glob import glob
import json
from utils import *
import pandas as pd

dataset_path = '../data'
original_path = '../original_data'
train_path = f'{original_path}/training/*'
# train_path = f'{original_path}/concept_train/*'
train_files = glob(train_path)

test_path = f'{original_path}/evaluation/*'
# test_path = f'{original_path}/concept_eval/*'
test_files = glob(test_path)

arc_task_sample = f'{original_path}/arc_task_sample2.csv'
df_arc_task_sample = pd.read_csv(arc_task_sample, encoding='utf-8')

train_dataset = {}
valid_dataset = {}
test_dataset = {}

train_input = []
train_output = []
train_input_size = []
train_output_size = []
train_task_class = []

valid_input = []
valid_output = []
valid_input_size = []
valid_output_size = []
valid_task_class = []

test_input = []
test_output = []
test_input_size = []
test_output_size = []
test_task_class = []

max_len = 30
count = 0

SAMPLE_FLAG = False
BC_FLAG = True
SOFT_FLAG = False

for file in train_files:
    with open(file, 'rb') as f:
        data = json.load(f)
    train_data = data['train']
    valid_data = data['test']
    file_name = file.split('\\')[-1].split('.')[0]
    if not df_arc_task_sample[df_arc_task_sample['Name'] == file_name]['tag'].tolist():
        continue
    if BC_FLAG:
        target_file_name = df_arc_task_sample[df_arc_task_sample['Name'] == file_name]['class'].tolist()[0].split(', ')
    else:
        target_file_name = df_arc_task_sample[df_arc_task_sample['Name'] == file_name]['tag'].tolist()[0]

    for i in range(len(train_data)):
        train_data[i]['input'] = (np.array(train_data[i]['input'])).tolist()
        train_input_y_len, train_input_x_len = np.array(train_data[i]['input']).shape
        train_output_y_len, train_output_x_len = np.array(train_data[i]['output']).shape
        input_size = (train_input_y_len, train_input_x_len)
        output_size = (train_output_y_len, train_output_x_len)

        input_arr = [[0 if m > train_input_x_len-1 or n > train_input_y_len-1 else train_data[i]['input'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
        output_arr = [[0 if m > train_output_x_len - 1 or n > train_output_y_len - 1 else train_data[i]['output'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
        if True in [11 in input_arr[x] for x in range(30)] or True in [11 in output_arr[x] for x in range(30)]:
            pass
        train_input.append(input_arr)
        train_output.append(output_arr)

        train_input_size.append(input_size)
        train_output_size.append(output_size)

        train_task_class.append(target_file_name)


    for i in range(len(valid_data)):
        valid_data[i]['input'] = (np.array(valid_data[i]['input'])).tolist()
        valid_input_y_len, valid_input_x_len = np.array(valid_data[i]['input']).shape
        valid_output_y_len, valid_output_x_len = np.array(valid_data[i]['output']).shape

        input_size = (valid_input_y_len, valid_input_x_len)
        output_size = (valid_output_y_len, valid_output_x_len)

        input_arr = [[0 if m > valid_input_x_len - 1 or n > valid_input_y_len - 1 else valid_data[i]['input'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
        output_arr = [ [0 if m > valid_output_x_len - 1 or n > valid_output_y_len - 1 else valid_data[i]['output'][n][m]+1 for m in range(max_len)] for n in range(max_len)]

        if True in [11 in input_arr[x] for x in range(30)] or True in [11 in output_arr[x] for x in range(30)]:
            pass

        # train_input.append(input_arr)
        # train_output.append(output_arr)
        #
        # train_input_size.append(input_size)
        # train_output_size.append(output_size)
        #
        # train_task_class.append(file_name)

        valid_input.append(input_arr)
        valid_output.append(output_arr)

        valid_input_size.append(input_size)
        valid_output_size.append(output_size)

        valid_task_class.append(target_file_name)

    # count += 1
    # if count == 10:
    #     break
    if SAMPLE_FLAG:
        print(file)
        break


for file in test_files:
    with open(file, 'rb') as f:
        data = json.load(f)
    train_data = data['train']
    valid_data = data['test']
    file_name = file.split('\\')[-1].split('.')[0][:-2]

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
        test_output.append(output_arr)

        test_input_size.append(input_size)
        test_output_size.append(output_size)

        test_task_class.append(file_name)

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
        test_task_class.append(file_name)

train_json_data = {
    'input': train_input,
    'output': train_output,
    'input_size': train_input_size,
    'output_size': train_output_size,
    'task': train_task_class,
}

valid_json_data = {
    'input': valid_input,
    'output': valid_output,
    'input_size': valid_input_size,
    'output_size': valid_output_size,
    'task': valid_task_class,
}

test_json_data = {
    'input': test_input,
    'output': test_output,
    'input_size': test_input_size,
    'output_size': test_output_size,
    'task': test_task_class,
}

# train_dataframe = pd.DataFrame({
#     'input': train_input,
#     'output': train_output
# })
#
# valid_dataframe = pd.DataFrame({
#     'input': valid_input,
#     'output': valid_output
# })

# train_dataframe.to_csv('arc_train.csv', index=None)
# valid_dataframe.to_csv('arc_valid.csv', index=None)
# a = pd.read_csv('arc_train.csv')
# b = pd.read_csv('arc_valid.csv')





with open(f'{dataset_path}/train_new_idea_task_sample2.json', 'w') as f:
    json.dump(train_json_data,f)

with open(f'{dataset_path}/valid_new_idea_task_sample2.json', 'w') as f:
    json.dump(valid_json_data, f)

# with open(f'{dataset_path}/test_new_idea_concept.json', 'w') as f:
#     json.dump(test_json_data, f)

