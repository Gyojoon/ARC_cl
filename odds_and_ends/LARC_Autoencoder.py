from transformers import AutoTokenizer, T5EncoderModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import numpy as np
import pandas as pd
import grid_visual
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


class Grid_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, , 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        feature_map = self.encoder(x)

        return feature_map

class prototype_fusion_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Grid_Encoder()
        self.encoder.state_dict(torch.load('auto_200.pt'))
        self.t5_encoder = T5EncoderModel.from_pretrained("t5-small")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

        self.resize_layer1 = nn.Linear(484,512)
        self.resize_layer2 = nn.Linear(120, 20)
        self.fusion_layer = nn.Linear(512,484)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

        self.model_freeze(self.encoder)
        self.model_freeze(self.t5_encoder)


    def model_freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x, LARC_description):
        # ========== Encoder ==========
        feature_map = self.encoder(x)
        feature_map = feature_map.view(20, -1)
        resize_feature_map = self.resize_layer1(feature_map)
        # ========== T5 Encoder ==========
        input_ids = self.t5_tokenizer(LARC_description, return_tensors="pt", padding='max_length', max_length=100, truncation=True,).input_ids.to('cuda') # Batch size 1
        outputs = self.t5_encoder(input_ids=input_ids)
        LARC_representation = outputs.last_hidden_state.squeeze()

        # ========== concat & fusion layer ==========
        concat_feature = torch.cat((resize_feature_map, LARC_representation))
        fustion_feature = self.fusion_layer(concat_feature)

        # ========== CNN decoder ==========
        resize_fusion_feature1 = fustion_feature.view(-1, 22, 22)
        permute_resize_fusion_feature = resize_fusion_feature1.permute(1,2,0)
        resize_fusion_feature2 = self.resize_layer2(permute_resize_fusion_feature)
        final_resize = resize_fusion_feature2.permute(2,0,1)

        result = self.decoder(final_resize)

        return result



#TODO LARD_Dataset class 구현하기
class LARC_Dataset(Dataset):
  def __init__(self, grid_files, LARC_file_name):
    self.grid_files = grid_files
    self.LARC_dataset = None
    with open(LARC_file_name, 'r') as f:
        self.LARC_dataset = json.load(f)

  def __len__(self):
    return len(self.LARC_dataset['task_name'])


  def __getitem__(self,idx):
    grid_file = self.grid_files[idx]
    task_name = self.LARC_dataset['task_name'][idx]
    task_description_output = self.LARC_dataset['description_output'][idx]
    return grid_file, task_name, task_description_output

with open('LARC_data.json', 'r') as f:
    larc_data = json.load(f)

# Task 312는 LARC에서 description_output들이 모두 NaN이라서 사용하지 않을 예정

grid_files = glob.glob('training/*')
grid_files.pop(312)
LARC_file_name = 'LARC_data.json'
batch_size = 1
mode = 'train'
use_valid = True
max_len = 30
epochs = 1
lr = 1e-5
use_pretrain = True
pretrain = 'LARC_200.pt'

#TODO 아래의 두개의 모델을 하나의 클래스로 담아서 해당 클래스의 객체를 model instance로 만들어서 optimizer 사용해야 함.
grid_encoder = Grid_Encoder().to('cuda')
grid_encoder.load_state_dict(torch.load('auto_200.pt'))
tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_encoder = T5EncoderModel.from_pretrained("t5-small")

model = prototype_fusion_model().to('cuda')

optimizer = optim.AdamW(model.parameters(), lr=lr)
criteria = nn.MSELoss().to('cuda')
pbar = tqdm(range(1,epochs+1))

train_LARC_dataset = LARC_Dataset(grid_files, LARC_file_name)
valid_LARC_dataset = LARC_Dataset(grid_files, LARC_file_name)


train_loader = DataLoader(train_LARC_dataset, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_LARC_dataset, batch_size=batch_size, drop_last=True)

pbar = tqdm(range(1,epochs+1))


# LARC description_input과 description_output의 최대 토큰 개수는 31

for epoch in pbar:
    if use_pretrain:
        model.load_state_dict(torch.load(pretrain))
    else:
        train_loss = []
        for grid_file_name, task_name, task_description_output in train_loader:
            if not task_name[0] in grid_file_name[0]:
                raise('grid_file_name과 LARC task name이 일치하지 않음')
            if mode == 'train':
                with open(grid_file_name[0], 'r') as f:
                    data = json.load(f)['train']
            for i in range(len(data)):
                input_data = (np.array(data[i]['input']) + 1).tolist()
                output_data = (np.array(data[i]['output']) + 1).tolist()
                train_input_y_len, train_input_x_len = np.array(input_data).shape
                train_output_y_len, train_output_x_len = np.array(output_data).shape
                input_size = (train_input_y_len, train_input_x_len)
                output_size = (train_output_y_len, train_output_x_len)
                train_input = [[0 if m > train_input_x_len - 1 or n > train_input_y_len - 1 else data[i]['input'][n][m] + 1 for m in range(max_len)] for n in range(max_len)]
                train_output = [[0 if m > train_output_x_len - 1 or n > train_output_y_len - 1 else data[i]['output'][n][m] + 1 for m in range(max_len)] for n in range(max_len)]

                train_input = torch.tensor(train_input, dtype=torch.float32).to('cuda').unsqueeze(0)
                train_output = torch.tensor(train_output, dtype=torch.float32).to('cuda').unsqueeze(0)

                output = model(train_input, task_description_output)

                label_grid = train_output[:, :output_size[0], :output_size[1]].detach().cpu() - 1
                output_grid = torch.round(output[:, :output_size[0], :output_size[1]].detach().cpu()) -1

                loss = criteria(train_output[:, :output_size[0], :output_size[1]], output[:, :output_size[0], :output_size[1]])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss)

        pbar.write(f'train_loss: {sum(train_loss) / len(train_loss)}')

    total_acc = 0
    total_loss = []
    count = 0

    if use_valid:
        for grid_file_name, task_name, task_description_output in valid_loader:
            with open(grid_file_name[0], 'r') as f:
                data = json.load(f)['test']
            for i in range(len(data)):
                try:
                    input_data = (np.array(data[i]['input']) + 1).tolist()
                except:
                    print(epoch)
                output_data = (np.array(data[i]['output']) + 1).tolist()
                valid_input_y_len, valid_input_x_len = np.array(input_data).shape
                valid_output_y_len, valid_output_x_len = np.array(output_data).shape
                input_size = (valid_input_y_len, valid_input_x_len)
                output_size = (valid_output_y_len, valid_output_x_len)
                valid_input = [
                    [0 if m > valid_input_x_len - 1 or n > valid_input_y_len - 1 else data[i]['input'][n][m] + 1 for
                     m in range(max_len)] for n in range(max_len)]
                valid_output = [
                    [0 if m > valid_output_x_len - 1 or n > valid_output_y_len - 1 else data[i]['output'][n][m] + 1
                     for m in range(max_len)] for n in range(max_len)]

                train_input = torch.tensor(valid_input, dtype=torch.float32).to('cuda').unsqueeze(0)
                train_output = torch.tensor(valid_output, dtype=torch.float32).to('cuda').unsqueeze(0)

                output = model(train_input, task_description_output)

                label_grid = train_output[:, :output_size[0], :output_size[1]].detach().cpu() - 1
                output_grid = torch.round(output[:, :output_size[0], :output_size[1]].detach().cpu()) - 1

                # output = output[:,:size[0][0], :size[0][1]].detach().cpu()
                round_output = torch.round(output[:, :output_size[0], :output_size[1]]) - 1
                if torch.equal(round_output, train_output[:, :output_size[0], :output_size[1]] - 1):
                    total_acc += 1
                else:
                    pass
                # total_loss.append(loss)
                count += 1
        pbar.write(f'acc: {total_acc / count}({total_acc}/{count})')

if not use_pretrain:
    torch.save(model.state_dict(), f'LARC_{epochs}.pt')