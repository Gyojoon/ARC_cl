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
    run = wandb.init(project=f'{dataset}_{target_name}_{mode}', entity='whatchang', )
    if mode == 'train':
        config = {
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': optimizer
        }
        wandb.config.update(config)
        wandb.run.name = f'{model_name}_o{optimizer}_b{batch_size}_e{epochs}_s{seed}_p{permute_mode}'
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

        #TODO Modulelist와 for문으로 다시 작성하기
        self.move_layer = nn.Linear(self.last_parameter_size, 1)
        self.color_layer = nn.Linear(self.last_parameter_size, 1)
        self.object_layer = nn.Linear(self.last_parameter_size, 1)
        self.pattern_layer = nn.Linear(self.last_parameter_size, 1)
        self.count_layer = nn.Linear(self.last_parameter_size, 1)
        self.crop_layer = nn.Linear(self.last_parameter_size, 1)
        self.boundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.center_layer = nn.Linear(self.last_parameter_size, 1)
        self.resie_layer = nn.Linear(self.last_parameter_size, 1)
        self.inside_layer = nn.Linear(self.last_parameter_size, 1)
        self.outside_layer = nn.Linear(self.last_parameter_size, 1)
        self.remove_layer = nn.Linear(self.last_parameter_size, 1)
        self.copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.position_layer = nn.Linear(self.last_parameter_size, 1)
        self.direction_layer = nn.Linear(self.last_parameter_size, 1)
        self.bitwise_layer = nn.Linear(self.last_parameter_size, 1)
        self.connect_layer = nn.Linear(self.last_parameter_size, 1)
        self.order_layer = nn.Linear(self.last_parameter_size, 1)
        self.combine_layer = nn.Linear(self.last_parameter_size, 1)
        self.fill_layer = nn.Linear(self.last_parameter_size, 1)

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

        # fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.norm_layer2(fusion_feature)
        # fusion_feature = self.relu(fusion_feature)
        #
        # fusion_feature = self.fusion_layer3(fusion_feature)
        # fusion_feature = self.norm_layer3(fusion_feature)
        # pre_fusion_feature = self.relu(fusion_feature)
        #
        # fusion_feature = self.fusion_layer4(fusion_feature)
        # fusion_feature = self.norm_layer3(fusion_feature)
        # fusion_feature += pre_fusion_feature
        # pre_fusion_feature = self.relu(fusion_feature)
        #
        # fusion_feature = self.fusion_layer5(fusion_feature)
        # fusion_feature = self.norm_layer3(fusion_feature)
        # fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)


        # output = self.task_proj(fusion_feature)


        # ===================== 경계선 ======================#


        move_output = self.move_layer(fusion_feature)
        color_output = self.color_layer(fusion_feature)
        object_output = self.object_layer(fusion_feature)
        pattern_output = self.pattern_layer(fusion_feature)
        count_output = self.count_layer(fusion_feature)
        crop_output = self.crop_layer(fusion_feature)
        boundary_output = self.boundary_layer(fusion_feature)
        center_output = self.center_layer(fusion_feature)
        resize_output = self.resie_layer(fusion_feature)
        inside_output = self.inside_layer(fusion_feature)
        outside_output = self.outside_layer(fusion_feature)
        remove_output = self.remove_layer(fusion_feature)
        copy_output = self.copy_layer(fusion_feature)
        position_output = self.position_layer(fusion_feature)
        direction_output = self.direction_layer(fusion_feature)
        bitwise_output = self.bitwise_layer(fusion_feature)
        connect_output = self.connect_layer(fusion_feature)
        order_output = self.order_layer(fusion_feature)
        combine_output = self.combine_layer(fusion_feature)
        fill_output = self.fill_layer(fusion_feature)

        output = torch.stack([move_output, color_output, object_output, pattern_output, count_output, crop_output, boundary_output, center_output, resize_output, inside_output, outside_output, remove_output, copy_output, position_output, direction_output, bitwise_output, connect_output, order_output, combine_output, fill_output])


        return output

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


        #TODO Modulelist와 for문으로 다시 작성하기
        self.move_layer = nn.Linear(self.last_parameter_size, 1)
        self.color_layer = nn.Linear(self.last_parameter_size, 1)
        self.object_layer = nn.Linear(self.last_parameter_size, 1)
        self.pattern_layer = nn.Linear(self.last_parameter_size, 1)
        self.count_layer = nn.Linear(self.last_parameter_size, 1)
        self.crop_layer = nn.Linear(self.last_parameter_size, 1)
        self.boundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.center_layer = nn.Linear(self.last_parameter_size, 1)
        self.resie_layer = nn.Linear(self.last_parameter_size, 1)
        self.inside_layer = nn.Linear(self.last_parameter_size, 1)
        self.outside_layer = nn.Linear(self.last_parameter_size, 1)
        self.remove_layer = nn.Linear(self.last_parameter_size, 1)
        self.copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.position_layer = nn.Linear(self.last_parameter_size, 1)
        self.direction_layer = nn.Linear(self.last_parameter_size, 1)
        self.bitwise_layer = nn.Linear(self.last_parameter_size, 1)
        self.connect_layer = nn.Linear(self.last_parameter_size, 1)
        self.order_layer = nn.Linear(self.last_parameter_size, 1)
        self.combine_layer = nn.Linear(self.last_parameter_size, 1)
        self.fill_layer = nn.Linear(self.last_parameter_size, 1)

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
        fusion_feature = self.leaky_relu(fusion_feature)

        # fusion_feature = self.fusion_layer2(fusion_feature)
        # # fusion_feature = self.dropout(fusion_feature)
        # fusion_feature = self.norm_layer2(fusion_feature)
        # # fusion_feature += pre_fusion_feature
        # # fusion_feature = self.relu(fusion_feature)
        # fusion_feature = self.leaky_relu(fusion_feature)



        # output = self.task_proj(fusion_feature)


        # ===================== 경계선 ======================#


        move_output = self.move_layer(fusion_feature)
        color_output = self.color_layer(fusion_feature)
        object_output = self.object_layer(fusion_feature)
        pattern_output = self.pattern_layer(fusion_feature)
        count_output = self.count_layer(fusion_feature)
        crop_output = self.crop_layer(fusion_feature)
        boundary_output = self.boundary_layer(fusion_feature)
        center_output = self.center_layer(fusion_feature)
        resize_output = self.resie_layer(fusion_feature)
        inside_output = self.inside_layer(fusion_feature)
        outside_output = self.outside_layer(fusion_feature)
        remove_output = self.remove_layer(fusion_feature)
        copy_output = self.copy_layer(fusion_feature)
        position_output = self.position_layer(fusion_feature)
        direction_output = self.direction_layer(fusion_feature)
        bitwise_output = self.bitwise_layer(fusion_feature)
        connect_output = self.connect_layer(fusion_feature)
        order_output = self.order_layer(fusion_feature)
        combine_output = self.combine_layer(fusion_feature)
        fill_output = self.fill_layer(fusion_feature)

        output = torch.stack([move_output, color_output, object_output, pattern_output, count_output, crop_output, boundary_output, center_output, resize_output, inside_output, outside_output, remove_output, copy_output, position_output, direction_output, bitwise_output, connect_output, order_output, combine_output, fill_output])


        return output

train_batch_size = 32
valid_batch_size = 1
lr = 5e-5
batch_size = train_batch_size
epochs = 400
seed = 777
use_wandb = True
mode = 'multi-bc'
model_name = 'vae'
permute_mode = True
use_scheduler = False
scheduler_name = 'lambda'
lr_lambda = 0.97

new_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')
train_dataset_name = 'data/train_new_idea_task_sample2.json'
valid_dataset_name = 'data/valid_new_idea_task_sample2.json'
train_dataset = New_ARCDataset(train_dataset_name, mode=mode, permute_mode=permute_mode)
valid_dataset = New_ARCDataset(valid_dataset_name, mode=mode)
kind_of_dataset = 'ARC_task_sample2' #'concept' if 'concept' in train_dataset_name else 'ARC'
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=True)

# optimizer = optim.AdamW(new_model.parameters(), lr=lr)
optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: lr_lambda ** epoch)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

if 'multi-bc' == mode:
    criteria = nn.BCELoss().to('cuda')
elif 'multi-soft' == mode:
    criteria = nn.BCEWithLogitsLoss(reduction='mean')
else:
    criteria = nn.CrossEntropyLoss().to('cuda')
seed_fix(777)

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
    for input, output, task in train_loader:
        train_count += train_batch_size
        input = input.to(torch.float32).to('cuda')
        output = output.to(torch.float32).to('cuda')
        if 'multi-soft' == mode:
            task = task.to(torch.float32).to('cuda')
        else:
            task = task.to(torch.long).to('cuda')
        output = new_model(input, output)

        if 'multi-bc' == mode:
            task_losses = []
            for i in range(task.shape[1]):
                # loss = criteria(nn.functional.softmax(output).permute(1,0,2)[i], task[i].to(torch.float32))
                loss = criteria(nn.functional.sigmoid(output), task.permute(1,0,2).to(torch.float32))
                # task_losses.append(loss)
            # torch.mean(task_losses).backward()
        else:
            loss = criteria(output, task)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_total_loss.append(loss)
        if mode == 'multi-bc':
            temp_list = []
            for i in range(train_batch_size):
                temp_list.append(torch.where(torch.round(nn.functional.sigmoid(output.permute(1, 0, 2))[i]).eq(task[i]).sum() == 20,1, 0))
            train_total_acc += torch.tensor(temp_list).sum()
            # train_total_acc += output_argmax.permute(1,0,2).eq(task).sum().item()
        elif mode =='multi-soft':
            temp_list = []
            for i in range(train_batch_size):
                attribute_count = torch.where(task[0] == 0, 0, 1).sum()
                threshold_point = torch.round(1 / attribute_count, decimals=2)
                pred = torch.where(nn.functional.threshold(nn.functional.softmax(output[i], dim=0),torch.round(1 / (attribute_count + 2), decimals=2), 0) == 0, 0, 1)
                answer = torch.where(task[i]==0, 0, 1)
                correct_check = torch.where(answer.eq(pred).sum() == 20, 1, 0)
                # torch.where(torch.where(task[i]==0, 0, 1).eq(torch.where(nn.functional.threshold(nn.functional.softmax(output.permute(1, 0, 2)[i], dim=0),torch.round(1 / (attribute_count + 2), decimals=2), 0) == 0, 0, 1)).sum() == 20, 1, 0)
                # nn.functional.threshold(nn.functional.softmax(output.permute(1,0,2)[i],dim=0), torch.round(1 / (attribute_count+2), decimals=2), 0)
                temp_list.append(correct_check)
            train_total_acc += torch.tensor(temp_list).sum()
        else:
            output_softmax= torch.softmax(output,dim=1)
            output_argmax = torch.argmax(output_softmax, dim=1)
            train_total_acc += output_argmax.eq(task).sum().item()
        # if output_argmax == task:
        #     train_total_acc += 1

    if use_scheduler:
        scheduler.step(sum(valid_total_loss) / len(valid_total_loss))
    print(f'train loss: {sum(train_total_loss) / len(train_total_loss)}')
    if 'multi' in mode:
        print(f'train acc: {train_total_acc / train_count}({train_total_acc}/{train_count})')
    else:
        print(f'train acc: {train_total_acc / train_count}({train_total_acc}/{train_count})')
    if use_wandb:
        wandb.log({
            "train_loss": sum(train_total_loss) / len(train_total_loss),
            "train_acc": train_total_acc / train_count,
            "train_correct_num": train_total_acc,
        }, step=epoch)

    new_model.eval()
    for input, output, task in valid_loader:
        valid_count += valid_batch_size
        input = input.to(torch.float32).to('cuda')
        output = output.to(torch.float32).to('cuda')
        if 'multi-soft' == mode:
            task = task.to(torch.float32).to('cuda')
        else:
            task = task.to(torch.long).to('cuda')
        output = new_model(input, output)

        if 'multi-bc' == mode:
            loss = criteria(nn.functional.sigmoid(output), task.permute(1,0,2).to(torch.float32))
        else:
            loss = criteria(output, task)
        valid_total_loss.append(loss)

        if mode == 'multi-bc':
            temp_list = []
            for i in range(valid_batch_size):
                temp_list.append(torch.where(torch.round(nn.functional.sigmoid(output.permute(1,0,2))[i]).eq(task[i]).sum() == 20,1, 0))
            valid_total_acc += torch.tensor(temp_list).sum()
            # valid_total_acc += torch.round(nn.functional.sigmoid(output.permute(1,0,2))).eq(task)
        elif mode == 'multi-soft':
            temp_list = []
            for i in range(valid_batch_size):
                attribute_count = torch.where(task[0] == 0, 0, 1).sum()
                threshold_point = torch.round(1 / attribute_count, decimals=2)
                pred = torch.where(nn.functional.threshold(nn.functional.softmax(output[i], dim=0),torch.round(1 / (attribute_count + 2), decimals=2), 0) == 0, 0, 1)
                answer = torch.where(task[i] == 0, 0, 1)
                correct_check = torch.where(answer.eq(pred).sum() == 20, 1, 0)
                temp_list.append(correct_check)
            valid_total_acc += torch.tensor(temp_list).sum()
        else:
            output_softmax = torch.softmax(output, dim=1)
            output_argmax = torch.argmax(output_softmax, dim=1)
            valid_total_acc += output_argmax.eq(task).sum().item()
        # if output_argmax == task:
        #     valid_total_acc += 1
    print(f'valid loss: {sum(valid_total_loss) / len(valid_total_loss)}')
    if 'multi' in mode:
        print(f'valid acc: {valid_total_acc / valid_count}({valid_total_acc}/{valid_count})')
    else:
        print(f'valid acc: {valid_total_acc / valid_count}({valid_total_acc}/{valid_count})')

    if use_wandb:
        wandb.log({
            "valid_loss": sum(valid_total_loss) / len(valid_total_loss),
            "valid_acc": valid_total_acc / valid_count,
            "valid_correct_num": valid_total_acc,
        }, step=epoch)