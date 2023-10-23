import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ARCDataset, ARC_ValidDataset
from tqdm import tqdm
import torch.optim as optim
from lion_pytorch import Lion
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import yaml
from loss import *
from utils import *
from model import *

with open('hyper/CL_concept_classifier.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

entity = config['entity']
permute_mode = config['permute_mode']
train_batch_size = config['train_batch_size']
valid_batch_size = config['valid_batch_size']
lr = float(config['lr'])
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
seed_fix(seed)
early_stopping = EarlyStopping(patience=40, verbose=True, path='best_concept_classifier_model.pt')  # 초기화
#lr_lambda = 0.97

new_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')         #Cross_vae_Linear_origin_b64_lr1e-3_4.pt이게 뭔지 확인!
# train_dataset_name = 'data/train_new_idea.json'
# valid_dataset_name = 'data/valid_new_idea.json'
train_dataset_name = 'data/train_concept.json'
valid_dataset_name = 'data/test_concept.json'
# train_dataset_name = 'data/train_new_idea_task_sample2_.json'
# valid_dataset_name = 'data/valid_new_idea_task_sample2_.json'
train_dataset = ARCDataset(train_dataset_name, mode=mode, permute_mode=permute_mode)
valid_dataset = ARC_ValidDataset(valid_dataset_name, mode=mode)
kind_of_dataset = 'Concept_task_sample2' if 'concept' in train_dataset_name else 'ARC_task_sample2' if 'sample2' in train_dataset_name else 'ARC'
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=False)

#optimizer = optim.AdamW(new_model.parameters(), lr=lr)
#optimizer = optim.Adam(new_model.parameters(), lr=lr, weight_decay=5e-4)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: lr_lambda ** epoch)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criteria = nn.CrossEntropyLoss()

best_acc = 0

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
print(f'Before KNN Accuracy: {accuracy * 100:.2f}%')


new_model.load_state_dict(torch.load('result\CL_soft_cl_0.2418329417705536.pt'))
for param in new_model.parameters():
    param.requires_grad = False
new_model.classifier = nn.Linear(128, 16).to('cuda')
optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, verbose=True)


if use_wandb:
    set_wandb('new_idea', 'train', seed, 'Lion', kind_of_dataset, entity)


def nt_xent_loss(output, temperature):
    batch_size = output.shape[0]
    logits = torch.mm(output, output.t().contiguous()) / temperature
    labels = torch.arange(batch_size).to(output.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

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
print(f'After KNN Accuracy: {accuracy * 100:.2f}%')

for epoch in tqdm(range(epochs)):
    train_total_loss = []
    train_total_acc = 0
    valid_total_loss = []
    valid_total_acc = 0
    train_count = 0
    valid_count = 0
    new_model.train()
    for input, output, x_size, y_size, task in train_loader:
        train_count += train_batch_size
        input = input.to(torch.float32).to('cuda')
        output = output.to(torch.float32).to('cuda')
        task = task.to(torch.long).to('cuda') # TODO 고치기

        output = new_model(input, output)
        output = new_model.classifier(output)

        loss = criteria(output, task)

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
        output = new_model.classifier(output)

        loss = criteria(output, task)

        valid_total_loss.append(loss) # To store the loss value and not the tensor
        output_soft = nn.functional.softmax(output, dim=1) # Specified dimension for softmax
        output_argmax = torch.argmax(output_soft, dim=1)   # Specified dimension for argmax

        acc += output_argmax.eq(task).sum() # Sum over the correct predictions

    avg_valid_loss = sum(valid_total_loss) / len(valid_total_loss)

    early_stopping(avg_valid_loss, new_model)
    early_stopping(-100 * acc/len(valid_dataset), new_model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    print(f'valid loss: {avg_valid_loss}')
    print(f'valid accuracy: {100 * acc/len(valid_dataset):.2f}% ({acc}/{len(valid_dataset)})')
    if best_acc < 100 * acc/len(valid_dataset):
        best_acc = 100 * acc/len(valid_dataset)

    if use_wandb:
        wandb.log({
            "valid_loss": avg_valid_loss,
        }, step=epoch)

    # if use_scheduler:
    #     scheduler.step(avg_valid_loss)

new_model.load_state_dict(torch.load('best_concept_classifier_model.pt'))
torch.save(new_model.state_dict(), f'result/concept_classifier_number_{best_acc:.2f}.pt')