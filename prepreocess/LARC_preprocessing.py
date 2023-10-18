import pandas as pd
import json
import numpy as np

path = '../original_data/LARC_dataset/summary/'
make_final_data = False

if not make_final_data:
    join_file = pd.read_csv(path+'join.csv')
    task_file = pd.read_csv(path+'task.csv')
    build_file = pd.read_csv(path+'build.csv')
    description_file = pd.read_csv(path+'description.csv')

    final_data = join_file.merge(task_file).merge(description_file)
    final_data = final_data[['task_id','task_name', 'description_output', 'confidence']]
    final_data.to_csv('filtering_LARC.csv', index=None)
    exit()
else:
    data = pd.read_csv('filtering_LARC.csv')
    data = data.dropna().reset_index().drop('index', axis=1)

pre_task_id = -1
pre_description = None
max_confidence = 0

task_id_list = []
task_name_list = []
description_output_list = []
confidence_list = []

for i in range(data.shape[0]):
    task_id = data['task_id'][i]
    task_name = data['task_name'][i]
    task_description = data['description_output'][i]
    confidence = data['confidence'][i]
    if pre_task_id != task_id:
        if pre_description != None:
            description_output_list.append(pre_description)
            confidence_list.append(max_confidence)
        pre_task_id = task_id
        task_id_list.append(task_id)
        task_name_list.append(task_name)
        pre_description = task_description
        max_confidence = confidence
    else:
        if confidence > max_confidence:
            max_confidence = confidence
            pre_description = task_description

description_output_list.append(pre_description)
confidence_list.append(max_confidence)

for i in range(400):
    if i not in task_id_list:
        print(i)

LARC_json = {
    'task_id': str(task_id_list),
    'task_name': task_name_list,
    'description_output': description_output_list,
    'confidence': str(confidence_list),
}

with open('../data/LARC_data.json', 'w') as f:
    json.dump(LARC_json, f)

print('ok')
