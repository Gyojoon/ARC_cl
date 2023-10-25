import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from utils import *
from tqdm import tqdm
from model import *
from dataset import *

parameter_len = 16
pretrained_batch_mocdel = 'result/concept_classifier_nx_xent_loss_22.91.pt'
pretrained_task_mocdel = 'result/concept_classifier_our_loss_22.91.pt'

batch_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda') 
batch_model.classifier = nn.Linear(128, parameter_len).to('cuda')
batch_model.load_state_dict(torch.load(f'{pretrained_batch_mocdel}'))
for param in batch_model.parameters():
    param.requires_grad = False

task_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda') 
task_model.classifier = nn.Linear(128, parameter_len).to('cuda')
task_model.load_state_dict(torch.load(f'{pretrained_batch_mocdel}'))
for param in task_model.parameters():
    param.requires_grad = False

valid_batch_size = 16
valid_dataset_name = 'data/test_concept.json'
valid_dataset = ARC_ValidDataset(valid_dataset_name)
dataset_dict = valid_dataset.task_dict
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=False)

batch_collect_input_list = []
batch_incollect_input_list = []
batch_collect_output_list = []
batch_incollect_output_list = []
batch_collect_task_list = []
batch_incollect_task_list = []
batch_incollect_label_list = []

task_collect_input_list = []
task_incollect_input_list = []
task_collect_output_list = []
task_incollect_output_list = []
task_collect_task_list = []
task_incollect_task_list = []
task_incollect_label_list = []

batch_model.eval()
task_model.eval()
for input, output, x_size, y_size, task in valid_loader:
    input = input.to('cuda')
    output = output.to('cuda')
    task = task.to(torch.long).to('cuda')  

    batch_output = batch_model(input, output)
    batch_output = batch_model.classifier(batch_output)

    task_output = task_model(input, output)
    task_output = task_model.classifier(task_output)

    batch_soft = nn.functional.softmax(batch_output, dim=1) 
    batch_argmax = torch.argmax(batch_soft, dim=1)  
    batch_collection_condition = batch_argmax.eq(task)
    batch_incollection_condition = batch_collection_condition == False
    batch_collect_task_list += task[batch_collection_condition].detach().cpu().numpy().tolist()
    batch_incollect_task_list += task[batch_incollection_condition].detach().cpu().numpy().tolist()
    batch_incollect_label_list += batch_argmax[batch_incollection_condition].detach().cpu().numpy().tolist()
    batch_collect_input_list += input[batch_collection_condition].detach().cpu().numpy().tolist()
    batch_incollect_input_list += input[batch_incollection_condition].detach().cpu().numpy().tolist()
    batch_collect_output_list += output[batch_collection_condition].detach().cpu().numpy().tolist()
    batch_incollect_output_list += output[batch_incollection_condition].detach().cpu().numpy().tolist()

    task_soft = nn.functional.softmax(task_output, dim=1) 
    task_argmax = torch.argmax(task_soft, dim=1)   
    task_collection_condition = task_argmax.eq(task)
    task_incollection_condition = task_collection_condition == False
    task_collect_task_list += task[task_collection_condition].detach().cpu().numpy().tolist()
    task_incollect_task_list += task[task_incollection_condition].detach().cpu().numpy().tolist()
    task_incollect_label_list += task_argmax[task_incollection_condition].detach().cpu().numpy().tolist()
    task_collect_input_list += input[task_collection_condition].detach().cpu().numpy().tolist()
    task_incollect_input_list += input[task_incollection_condition].detach().cpu().numpy().tolist()
    task_collect_output_list += output[task_collection_condition].detach().cpu().numpy().tolist()
    task_incollect_output_list += output[task_incollection_condition].detach().cpu().numpy().tolist()



html = plot_2d_grid(batch_collect_task_list, batch_collect_task_list, batch_collect_input_list, batch_collect_output_list, dataset_dict)
write_file(plot_html=html, dataset='concept', loss='nx_xent', mode='collect')

html = plot_2d_grid(batch_incollect_task_list, batch_incollect_label_list, batch_incollect_input_list, batch_incollect_output_list, dataset_dict)
write_file(plot_html=html, dataset='concept', loss='nx_xent', mode='incollect')

html = plot_2d_grid(task_collect_task_list, task_collect_task_list, task_collect_input_list, task_collect_output_list, dataset_dict)
write_file(plot_html=html, dataset='concept', loss='our', mode='collect')

html = plot_2d_grid(task_incollect_task_list, task_incollect_label_list, task_incollect_input_list, task_incollect_output_list, dataset_dict)
write_file(plot_html=html, dataset='concept', loss='our', mode='incollect')