import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ARCDataset
from tqdm import tqdm
import torch.optim as optim
from lion_pytorch import Lion
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import optuna
from optuna.samplers import TPESampler
import yaml
from loss import *
from utils import *
from model import *

with open('Search_hyper_CL.yaml', 'r') as f:
    config = yaml.loac(f, Loader=yaml.FullLoader)

entity = config['entity']
permute_mode = config['permute_mode']
train_batch_size = config['train_batch_size']
valid_batch_size = config['valid_batch_size']
lr = config['lr']
batch_size = train_batch_size
epochs = config['epochs']
seed = config['seed']
model_name = config['model_name']
mode = config['mode']
temperature = config['temperature']
use_wandb = config['use_wandb']
use_scheduler = config['use_scheduler']
scheduler_name = config['scheduler_name']
patience = config['patience']
loss_mode = config['loss_mode']

def objective(trial):
    new_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda') 
    best_acc = 0
    train_batch_size = 128
    valid_batch_size = 16
    epochs = 200
    seed = 777

    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    temperature = trial.suggest_float('temperature', 1e-1, 2)

    optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, verbose=True)
    
    if use_wandb:
        set_wandb(epochs, 'train', seed, 'Lion', train_batch_size, valid_batch_size, lr, temperature, kind_of_dataset, entity)

    for epoch in tqdm(range(epochs)):
        train_total_loss = []
        train_total_acc = 0
        valid_total_loss = []
        valid_total_acc = 0
        train_count = 0
        valid_count = 0
        new_model.train()
        acc = 0
        for input, output, x_size, y_size, task in train_loader:
            train_count += train_batch_size
            input = input.to(torch.float32).to('cuda')
            output = output.to(torch.float32).to('cuda')
            task = task.to(torch.long).to('cuda') # TODO 고치기

            output = new_model(input, output)

            labels = label_making(task).to('cuda')

            if loss_mode == 'nx_xent_loss':
                loss = nt_xent_loss(output, temperature)        #NT-Xent Loss 사용
            elif loss_mode == 'our_loss':
                loss = our_loss(output, labels, temperature) 

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_total_loss.append(loss)
        print(f'train loss: {sum(train_total_loss) / len(train_total_loss)}')
        # print("train loss: {0}, lr: {1:.6f}".format(sum(train_total_loss) / len(train_total_loss), optimizer.param_groups[0]['lr']))
        if use_scheduler:
            scheduler.step()

        if use_wandb:
            wandb.log({
                "train_loss": sum(train_total_loss) / len(train_total_loss),
            }, step=epoch)

        acc = 0
        new_model.eval()
        for input, output, x_size, y_size, task in valid_loader:
            valid_count += valid_batch_size
            input = input.to('cuda')
            output = output.to('cuda')
            task = task.to(torch.long).to('cuda')  # Moved the task tensor to the GPU as well

            output = new_model(input, output)

            labels = label_making(task).to('cuda')

            if loss_mode == 'nx_xent_loss':
                loss = nt_xent_loss(output, temperature)        #NT-Xent Loss 사용
            elif loss_mode == 'our_loss':
                loss = our_loss(output, labels, temperature) 

            valid_total_loss.append(loss) # To store the loss value and not the tensor

        avg_valid_loss = sum(valid_total_loss) / len(valid_total_loss)

        print(f'valid loss: {avg_valid_loss}')

        if use_wandb:
            wandb.log({
                "valid_loss": avg_valid_loss,
            }, step=epoch)
    
    wandb.finish()
    
    return avg_valid_loss

early_stopping = EarlyStopping(patience=patience, verbose=True, path='best_model.pt')  # 초기화
#lr_lambda = 0.97

new_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')         #Cross_vae_Linear_origin_b64_lr1e-3_4.pt이게 뭔지 확인!
#train_dataset_name = 'data/train_data.json'[
#valid_dataset_name = 'data/valid_data.json'
train_dataset_name = 'data/train_new_idea.json'
valid_dataset_name = 'data/valid_new_idea.json'
# train_dataset_name = 'data/train_new_idea_task_sample2_.json'
# valid_dataset_name = 'data/valid_new_idea_task_sample2_.json'
train_dataset = ARCDataset(train_dataset_name, mode=mode, permute_mode=permute_mode)
valid_dataset = ARCDataset(valid_dataset_name, mode=mode)
kind_of_dataset = 'Concept_task_sample2' if 'concept' in train_dataset_name else 'ARC_task_sample2' if 'sample2' in train_dataset_name else 'ARC'
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=True)

#optimizer = optim.AdamW(new_model.parameters(), lr=lr)
optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
#optimizer = optim.Adam(new_model.parameters(), lr=lr, weight_decay=5e-4)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: lr_lambda ** epoch)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# KNN 모델 초기화
knn_model = KNeighborsClassifier(n_neighbors=5)

train_embeddings = []  # 학습 데이터의 임베딩을 저장하기 위한 리스트
train_labels = []      # 학습 데이터의 레이블을 저장하기 위한 리스트

new_model.eval()
for input, output, x_size, y_size, task in train_loader:
    input = input.to(torch.float32).to('cuda')
    output = output.to(torch.float32).to('cuda')
    task = task.to(torch.long)
    
    with torch.no_grad():
        embeddings = new_model(input, output)
    train_embeddings.append(embeddings.cpu().numpy())
    train_labels.extend(task.numpy())

# 임베딩과 레이블의 형태를 조절합니다.
train_embeddings = np.vstack(train_embeddings)
train_labels = np.array(train_labels)

# KNN 모델 학습
knn_model.fit(train_embeddings, train_labels)

valid_embeddings = []  # 검증 데이터의 임베딩을 저장하기 위한 리스트
valid_labels = []      # 검증 데이터의 레이블을 저장하기 위한 리스트
for input, output, x_size, y_size, task in valid_loader:
    input = input.to(torch.float32).to('cuda')
    output = output.to(torch.float32).to('cuda')
    task = task.to(torch.long)
    
    with torch.no_grad():
        embeddings = new_model(input, output)
    valid_embeddings.append(embeddings.cpu().numpy())
    valid_labels.extend(task.numpy())

# 임베딩과 레이블의 형태를 조절합니다.
valid_embeddings = np.vstack(valid_embeddings)
valid_labels = np.array(valid_labels)

# KNN의 정확도를 계산합니다.
accuracy = knn_model.score(valid_embeddings, valid_labels)
print(f'KNN Accuracy: {accuracy * 100:.2f}%')

# sampler = SkoptSampler()
sampler = TPESampler(**TPESampler.hyperopt_parameters())
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=1000)

# new_model.load_state_dict(torch.load('best_model.pt'))
# torch.save(new_model.state_dict(), f'result/number1.pt')