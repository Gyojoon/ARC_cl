import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import New_ARCDataset
from tqdm import tqdm
import torch.optim as optim
from lion_pytorch import Lion
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np


def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def set_wandb(target_name, mode, seed, optimizer, dataset='ARC'):
    run = wandb.init(project=f'{dataset}_{target_name}_{mode}', entity='gyojoongu')
    if mode == 'train':
        config = {
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': optimizer
        }
        wandb.config.update(config)
        wandb.run.name = f'{model_name}_o{optimizer}_l{lr}_b{batch_size}_e{epochs}_s{seed}'
    wandb.run.save()
    return run

class Autoencoder_batch1_c10(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.proj = nn.Linear(10, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        output = self.decoder(feature_map)

        return output

class vae_Linear_origin(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(11, 512)

        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(128, 128)
        self.sigma_layer = nn.Linear(128, 128)

        self.proj = nn.Linear(512, 11)

    def forward(self, x):
        if len(x.shape) > 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.sigma_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps
        output = self.decoder(latent_vector)
        output = self.proj(output).reshape(-1,30,30,11).permute(0,3,1,2)

        return output

class new_idea_ae(nn.Module):
    def __init__(self, model_file):
        super().__init__()
        self.autoencoder = Autoencoder_batch1_c10()
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.last_parameter_size = 20*22*22

        self.fusion_layer1 = nn.Linear(2*20*22*22, 20*22*22)
        self.fusion_layer2 = nn.Linear(20*22*22, 10*22*22)
        self.fusion_layer3 = nn.Linear(10*22*22, 22*22)

        self.fusion_layer4 = nn.Linear(22 * 22, 22 * 22)
        self.fusion_layer5 = nn.Linear(22 * 22, 22 * 22)

        self.task_proj = nn.Linear(22*22, 20)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.norm_layer1 = nn.BatchNorm1d(20*22*22)
        self.norm_layer2 = nn.BatchNorm1d(10*22*22)
        self.norm_layer3 = nn.BatchNorm1d(22*22)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]
        input_x = input_x.unsqueeze(1)
        output_x = output_x.unsqueeze(1)
        input_feature = self.autoencoder.encoder(input_x)
        output_feature = self.autoencoder.encoder(output_x)

        concat_feature = torch.stack((input_feature, output_feature),dim=1)

        fusion_feature = self.fusion_layer1(concat_feature.reshape(batch_size, -1))
        # fusion_feature = self.dropout(fusion_feature)
        # fusion_feature = self.norm_layer1(fusion_feature)
        fusion_feature = self.relu(fusion_feature)

class new_idea_vae(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.autoencoder = vae_Linear_origin()
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.first_layer_parameter_size = 128
        self.second_layer_parameter_size = 128
        self.third_layer_parameter_size = 64
        self.last_parameter_size = 128
        self.num_categories = 16

        self.fusion_layer1 = nn.Linear(2*128*900, self.first_layer_parameter_size)
        self.fusion_layer2 = nn.Linear(self.first_layer_parameter_size, self.second_layer_parameter_size)
        self.fusion_layer3 = nn.Linear(self.second_layer_parameter_size, self.third_layer_parameter_size)

        # self.task_proj = nn.Linear(128, 20)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.norm_layer1 = nn.BatchNorm1d(self.first_layer_parameter_size)
        self.norm_layer2 = nn.BatchNorm1d(self.second_layer_parameter_size)
        self.norm_layer3 = nn.BatchNorm1d(self.third_layer_parameter_size)



    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]
        if len(input_x.shape) > 3:
            batch_size = input_x.shape[0]
            embed_input = self.autoencoder.embedding(input_x.reshape(batch_size, 900).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_input = self.autoencoder.embedding(input_x.reshape(-1, 900).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(-1, 900).to(torch.long))
        input_feature = self.autoencoder.encoder(embed_input)
        output_feature = self.autoencoder.encoder(embed_output)

        input_mu = self.autoencoder.mu_layer(input_feature)
        input_sigma = self.autoencoder.sigma_layer(input_feature)
        intput_std = torch.exp(0.5 * input_sigma)
        input_eps = torch.randn_like(intput_std)
        input_latent_vector = input_mu + intput_std * input_eps

        output_mu = self.autoencoder.mu_layer(output_feature)
        output_sigma = self.autoencoder.sigma_layer(output_feature)
        output_std = torch.exp(0.5 * output_sigma)
        output_eps = torch.randn_like(output_std)
        output_latent_vector = output_mu + output_std * output_eps

        concat_feature = torch.cat((input_latent_vector, output_latent_vector),dim=2)

        fusion_feature = self.fusion_layer1(concat_feature.reshape(batch_size, -1))
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.norm_layer1(fusion_feature)
        # fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.norm_layer2(fusion_feature)
        # fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        pre_fusion_feature = self.leaky_relu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.norm_layer2(fusion_feature)
        fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        output = self.leaky_relu(fusion_feature)

        return output

def label_making(task):
    task = task.tolist()
    label_index_list = []
    label_list = []
    for i in range(len(task)):
        if i == 0 or task[i] not in label_list:
            label_index_list.append(task[i])


    label_list = [label_index_list.index(x) for x in task]


    return torch.tensor(label_list, dtype=torch.long)

def compute_latent_vector(feature, mu_layer, sigma_layer):
    mu = mu_layer(feature)
    sigma = sigma_layer(feature)
    std = torch.exp(0.5 * sigma)
    eps = torch.randn_like(std)
    return mu + std * eps

train_batch_size = 32
valid_batch_size = 32
lr = 5e-5
batch_size = train_batch_size
epochs = 400
seed = 777
model_name = 'vae'
mode = 'task'
temperature = 0.7
use_wandb = True

new_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')
train_dataset_name = 'data/train_new_idea_concept.json'
valid_dataset_name = 'data/valid_new_idea_concept.json'
# train_dataset_name = 'data/train_new_idea_task_sample2_.json'
# valid_dataset_name = 'data/valid_new_idea_task_sample2_.json'
train_dataset = New_ARCDataset(train_dataset_name, mode=mode)
valid_dataset = New_ARCDataset(valid_dataset_name, mode=mode)
kind_of_dataset = 'Concept_task_sample2' if 'concept' in train_dataset_name else 'ARC_task_sample2' if 'sample2' in train_dataset_name else 'ARC'
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=True)

#optimizer = optim.AdamW(new_model.parameters(), lr=lr)
optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)


seed_fix(seed)

if use_wandb:
    set_wandb('new_idea', 'train', seed, 'Lion', kind_of_dataset)

for epoch in tqdm(range(epochs)):
    train_total_loss = []
    train_total_acc = 0
    valid_total_loss = []
    valid_total_acc = 0
    train_count = 0
    valid_count = 0
    new_model.train()
    for epoch in tqdm(range(epochs)):
        train_total_loss = 0
        new_model.train()
        for input, output, task in train_loader:
            input, output = input.float().cuda(), output.float().cuda()
            task = task.long().cuda()

            output = new_model(input, output)
            output_norm = output / output.norm(dim=-1, keepdim=True)
            logits_pred = torch.matmul(output_norm, output_norm.T) * np.exp(temperature)
            labels = label_making(task).cuda()

            loss = nn.functional.nll_loss(nn.functional.log_softmax(logits_pred, dim=1), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_total_loss += loss.item()

        print(f'train loss: {train_total_loss / len(train_loader)}')

        if use_wandb:
            wandb.log({"train_loss": train_total_loss / len(train_loader)}, step=epoch)

        valid_total_loss = 0
        new_model.eval()
        with torch.no_grad():
            for input, output, task in valid_loader:
                input, output = input.float().cuda(), output.float().cuda()
                output = new_model(input, output)
                output_norm = output / output.norm(dim=-1, keepdim=True)
                logits_pred = torch.matmul(output_norm, output_norm.T) * np.exp(temperature)
                labels = label_making(task).cuda()
                loss = nn.functional.nll_loss(nn.functional.log_softmax(logits_pred, dim=1), labels)
                valid_total_loss += loss.item()

        print(f'valid loss: {valid_total_loss / len(valid_loader)}')

        if use_wandb:
            wandb.log({"valid_loss": valid_total_loss / len(valid_loader)}, step=epoch)

torch.save(new_model.state_dict(), f'result/number1.pt')