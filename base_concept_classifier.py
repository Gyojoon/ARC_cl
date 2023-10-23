import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ARCDataset, ARC_ValidDataset
from tqdm import tqdm
import torch.optim as optim
from lion_pytorch import Lion
import wandb
from sklearn.neighbors import KNeighborsClassifier
import yaml
from loss import *
from utils import *
from model import *

with open('hyper/base_concept_classifier.yaml', 'r') as f:
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
early_stopping = EarlyStopping(patience=patience, verbose=True, path='best_base_concept_classifier_model.pt')  # 초기화
seed_fix(seed)
#lr_lambda = 0.97

new_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')         #Cross_vae_Linear_origin_b64_lr1e-3_4.pt이게 뭔지 확인!
new_model.classifier = nn.Linear(128, 16).to('cuda')   
#train_dataset_name = 'data/train_data.json'
#valid_dataset_name = 'data/valid_data.json'
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
optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
#optimizer = optim.Adam(new_model.parameters(), lr=lr, weight_decay=5e-4)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: lr_lambda ** epoch)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, verbose=True)
criteria = nn.CrossEntropyLoss()

best_acc = 0

if use_wandb:
    set_wandb(epochs, mode, seed, 'Lion', train_batch_size, valid_batch_size, lr, temperature, kind_of_dataset, entity, project_name='CL_classifier')


for epoch in tqdm(range(epochs)):
    train_total_loss = []
    train_total_acc = 0
    valid_total_loss = []
    valid_total_acc = 0
    train_count = 0
    valid_count = 0
    acc = 0
    new_model.train()
    for input, output, x_size, y_size, task in train_loader:
        train_count += train_batch_size
        input = input.to(torch.float32).to('cuda')
        output = output.to(torch.float32).to('cuda')
        task = task.to(torch.long).to('cuda') # TODO 고치기

        output = new_model(input, output)
        output = new_model.classifier(output)

        output_soft = nn.functional.softmax(output, dim=1) # Specified dimension for softmax
        output_argmax = torch.argmax(output_soft, dim=1)   # Specified dimension for argmax

        loss = criteria(output, task)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_total_loss.append(loss)
    print(f'train loss: {sum(train_total_loss) / len(train_total_loss)}')
    # print("train loss: {0}, lr: {1:.6f}".format(sum(train_total_loss) / len(train_total_loss), optimizer.param_groups[0]['lr']))
    scheduler.step()

    if use_wandb:
        wandb.log({
            "train_loss": sum(train_total_loss) / len(train_total_loss),
            'train accuracy': 100 * acc/len(train_dataset),
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

    # early_stopping(avg_valid_loss, new_model)
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
            'valid accuracy': 100 * acc/len(valid_dataset),
            'best_valid': best_acc
        }, step=epoch)

    # if use_scheduler:
    #     scheduler.step(avg_valid_loss)

new_model.load_state_dict(torch.load('best_base_concept_classifier_model.pt'))
torch.save(new_model.state_dict(), f'result/base_concept_classifier_number_{best_acc:.2f}.pt')