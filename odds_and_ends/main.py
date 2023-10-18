import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import wandb
import time
from time import localtime

from model import *
from dataset import ARCDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Autoencoder_origin(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(30, 20, 5, padding=0),
        nn.ReLU(),
        nn.Conv2d(20, 10, 5, padding=0),
        nn.ReLU(),
        )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(10, 20, kernel_size = 5, stride = 1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(20, 30, kernel_size = 5, stride = 1, padding=0),
        nn.ReLU(),
        )

    # self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)

  def forward(self, x):
    feature_map = self.encoder(x)
    decoder_input = feature_map #+ self.action_vector
    output = self.decoder(decoder_input)

    return output

class Autoencoder_batch1(nn.Module):
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
            nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.proj = nn.Linear(1, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        output = self.decoder(feature_map)

        return output

class Action_batch1_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = None
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(1, 11)
        # ============================ version 2 ============================
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.leaky_relu = nn.LeakyReLU()
        self.last_layer = nn.Linear(10, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def forward(self, x):
        feature_map1 = self.autoencoder.encoder(x)
        new_feature_map1 = feature_map1 + self.action_vector
        output = self.action_decoder1(new_feature_map1)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        # output_size = output.shape
        # # output = output.view(output_size[0], -1)
        # output = output.view(output_size[0], output_size[0], -1)
        # output = self.action_linear1(output)
        # output = output.view(output_size)
        # output = self.relu(output)

        # output = self.leaky_relu(output)

        # ============================ layer ============================


        feature_map2 = self.autoencoder.decoder[0](feature_map1)
        new_feature_map2 = feature_map2 + output
        output = self.action_decoder2(new_feature_map2)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        # output_size = output.shape
        # # output = output.view(output_size[0], -1)
        # output = output.view(output_size[0], -1)
        # output = self.action_linear2(output)
        # output = output.view(output_size)
        # output = self.relu(output)
        # # output = self.leaky_relu(output)

        # ============================ layer ============================

        feature_map3 = self.autoencoder.decoder[2](feature_map2)
        new_feature_map3 = feature_map3 + output
        result = self.relu(new_feature_map3)
        # output = self.leaky_relu(new_feature_map3)
        # ============================ version 2 ============================
        # output_size = output.shape
        # output = output.view(output_size[0],  -1)
        # output = self.action_linear2(output)
        # output = output.view(output_size).squeeze()
        # output = self.relu(output)
        #
        # output = output.permute(1,2,0)
        # output = self.last_layer(output)
        # output = output.permute(2,0,1)
        # result = self.relu(output)
        # result = self.leaky_relu(output)

        return result

class Action_batch1_v2(nn.Module):
    def __init__(self, pre_traind='auto_200.pt'):
        super().__init__()
        self.autoencoder = Autoencoder_batch1().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pre_traind))
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.relu = nn.ReLU()
        # ============================ version 2 ============================
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.leaky_relu = nn.LeakyReLU()
        self.last_layer = nn.Linear(10, 1)

        for param in self.autoencoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        feature_map1 = self.autoencoder.encoder(x)
        new_feature_map1 = feature_map1 + self.action_vector
        output = self.action_decoder1(new_feature_map1)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear1(output)
        output = output.view(output_size)
        output = self.relu(output)

        # output = self.leaky_relu(output)

        # ============================ layer ============================


        feature_map2 = self.autoencoder.decoder[0](feature_map1)
        new_feature_map2 = feature_map2 + output
        output = self.action_decoder2(new_feature_map2)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        output_size = output.shape
        # output = output.view(output_size[0], -1)
        output = output.view(output_size[0], -1)
        output = self.action_linear2(output)
        output = output.view(output_size)
        output = self.relu(output)
        # output = self.leaky_relu(output)

        # ============================ layer ============================

        feature_map3 = self.autoencoder.decoder[2](feature_map2)
        new_feature_map3 = feature_map3 + output
        output = self.relu(new_feature_map3)
        # output = self.leaky_relu(new_feature_map3)
        # ============================ version 2 ============================
        output_size = output.shape
        output = output.view(output_size[0],  -1)
        output = self.action_linear2(output)
        output = output.view(output_size)#.squeeze()
        result = self.relu(output)

        # output = output.permute(1,2,0)
        # output = self.last_layer(output)
        # output = output.permute(2,0,1)
        # result = self.relu(output)
        # # result = self.leaky_relu(output)

        return result

class Action_batch1_v3(nn.Module):
    def __init__(self, pre_traind='auto_200.pt'):
        super().__init__()
        self.autoencoder = Autoencoder_batch1().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pre_traind))
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.feature_linear1 = nn.Linear(676, 676)
        self.feature_linear2 = nn.Linear(900, 900)
        self.fusion_layer1 = nn.Linear(2*676, 676)
        self.fusion_layer2 = nn.Linear(2 * 900, 900)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()


        for param in self.autoencoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        feature_map_origin = self.autoencoder.encoder(x)
        # new_feature_map1 = feature_map1# + self.action_vector
        output = self.action_decoder1(feature_map_origin)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear1(output)

        feature_map1 = self.autoencoder.decoder[0](feature_map_origin)
        feature1_size = feature_map1.shape
        feature1 = output.view(feature1_size[0], -1)
        feature1 = self.relu(feature1)

        fusion_feature1 = torch.cat((output,feature1),dim=1)
        output = self.fusion_layer1(fusion_feature1)
        output = self.relu(output)
        output = output.reshape(output_size)

        # ============================ layer1 ============================
        output = self.action_decoder2(output)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear2(output)

        feature2 = self.autoencoder.decoder[2](feature_map1)
        feature2_size = feature2.shape
        feature2 = output.view(feature2_size[0], -1)
        feature2 = self.relu(feature2)

        fusion_feature2 = torch.cat((output, feature2), dim=1)
        output = self.fusion_layer2(fusion_feature2)
        output = self.relu(output)
        result = output.reshape(output_size)

        return result

class Action_batch1_v4(nn.Module):
    def __init__(self, pre_traind='auto_200.pt'):
        super().__init__()
        self.autoencoder = Autoencoder_batch1().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pre_traind))
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.feature_linear1 = nn.Linear(676, 676)
        self.feature_linear2 = nn.Linear(900, 900)
        self.fusion_layer1 = nn.Linear(2*676, 676)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()


        for param in self.autoencoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        feature_map_origin = self.autoencoder.encoder(x)
        # new_feature_map1 = feature_map1# + self.action_vector
        output = self.action_decoder1(feature_map_origin)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear1(output)

        feature_map1 = self.autoencoder.decoder[0](feature_map_origin)
        feature1_size = feature_map1.shape
        feature1 = output.view(feature1_size[0], -1)
        feature1 = self.relu(feature1)

        fusion_feature1 = torch.cat((output,feature1),dim=1)
        output = self.fusion_layer1(fusion_feature1)
        output = self.relu(output)
        output = output.reshape(output_size)

        # ============================ layer1 ============================
        output = self.action_decoder2(output)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear2(output)
        result = output.reshape(output_size)

        return result




class Autoencoder2(nn.Module):
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
        nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0),
        nn.ReLU(),
        )

    self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
    self.proj = nn.Linear(1,11)

  def forward(self, x):
    feature_map = self.encoder(x)
    decoder_input = feature_map + self.action_vector
    output = self.decoder(decoder_input)

    return output

class ARCDataset(Dataset):
  def __init__(self, file_name, mode=None, permute_mode=False):
    self.dataset = None
    self.mode = mode
    self.permute_mode = permute_mode
    self.permute_color = np.random.choice(11, 11, replace=False)
    with open(file_name, 'r') as f:
      self.dataset = json.load(f)

  def __len__(self):
    if self.mode == 'Auto_encoder':
        return len(self.dataset['data'])
    else:
        return len(self.dataset['input'])

  def __getitem__(self,idx):
    if self.mode == 'Auto_encoder':
        x = self.dataset['data'][idx]
        size = self.dataset['size'][idx]
        if self.permute_mode:
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
        return torch.tensor(x), torch.tensor(size)
    else:
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]
        x_size = self.dataset['input_size'][idx]
        y_size = self.dataset['output_size'][idx]
        return x, y, x_size, y_size
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-7

    def forward(self,y,y_hat):
        return torch.sqrt(self.mse(y,y_hat) + self.eps)

def Auto_MSE_method(args, train_loader, valid_loader, test_loader):
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    use_permute = args.use_permute
    use_pretrain = args.use_pretrain
    use_valid = args.use_valid
    use_test = args.use_test
    use_batch = args.use_batch
    use_size = args.use_size
    use_wandb = args.use_wandb
    train_auto = args.train_auto
    pretrain = args.pretrain
    kind_of_loss = args.kind_of_loss

    model = Autoencoder_batch1().to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criteria = nn.MSELoss().to('cuda')

    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
    if use_permute:
        permute_color = np.random.choice(11, 11, replace=False)
    pbar = tqdm(range(1,epochs+1))
    for epoch in pbar:
        if use_pretrain:
            model.load_state_dict(torch.load(pretrain))
        else:
            train_loss = []
            for data, size in train_loader:
                data = torch.tensor(data).to(torch.float32).to('cuda')
                if use_batch:
                    data = data.unsqueeze(1)
                    # data = data.permute((1, 0, 2))


                output = model(data)
                x_grid_list = []
                output_grid_list = []

                if use_batch:
                    pass
                    # for i in range(batch_size):
                    #     x_grid_list.append(data[:, i, :].detach().cpu())
                    #     output_grid_list.append(data[:, i, :].detach().cpu())
                    # data_grid = torch.stack(x_grid_list).permute((1,0,2)) -1
                    # output_grid = torch.stack(output_grid_list).permute((1,0,2)) -1
                else:
                    data_grid = data[:, :size[0][0],:size[0][1]].detach().cpu() -1
                    output_grid = torch.round(output[:, :size[0][0], :size[0][1]].detach().cpu() -1)

                # plt.imshow(y_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
                # plt.imshow(output_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)

                if use_batch or not use_size:
                    # loss = criteria(data,output)
                    loss = criteria(output, data)
                else:
                    # loss = criteria(data[:, :size[0][0], :size[0][1]], output[:, :size[0][0], :size[0][1]])
                    loss = criteria(output[:, :size[0][0], :size[0][1]], data[:, :size[0][0], :size[0][1]])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss)
            scheduler.step()

            if use_wandb:
                wandb.log({
                    "train_loss": sum(train_loss) / len(train_loss),
                }, step=epoch)
            pbar.write(f'train_loss: {sum(train_loss) / len(train_loss)}')
            # print(f'train_loss: {sum(train_loss) / len(train_loss)}')

        total_acc = 0
        total_loss = []
        count = 0
        if use_valid:
            for data, size in valid_loader:
                data = torch.tensor(data).to(torch.float32).to('cuda')
                # if use_permute:
                #     for i in range(30):
                #         for j in range(30):
                #             data[:, i, j] = permute_color[data[:, i, j]]
                    # data = recursive_exchange(data, permute_color, 0, [])
                    # data = torch.where(data == -1, 0, data)
                output = model(data)

                data_grid = data[:, :size[0][0], :size[0][1]].squeeze().detach().cpu() - 1
                output_grid = torch.round(output[:,:size[0][0], :size[0][1]]).squeeze().detach().cpu() - 1

                # output = output[:,:size[0][0], :size[0][1]].detach().cpu()
                round_output = torch.round(output[:,:size[0][0], :size[0][1]]) -1

                if use_batch or not use_size:
                    loss = criteria(output, data)
                else:
                    loss = criteria(output[:, :, :size[0][0]*size[0][1]], data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))

                if torch.equal(round_output,data[:,:size[0][0], :size[0][1]] -1):
                  total_acc += 1
                else:
                    pass
                # total_loss.append(loss)
                count += 1

            if use_wandb:
                wandb.log({
                    "valid_loss": sum(train_loss) / len(train_loss),
                    "valid_acc" : f'acc: {total_acc/count}({total_acc}/{count})'
                }, step=epoch)
            pbar.write(f'acc: {total_acc/count}({total_acc}/{count})')

            if use_test:
                for data, size in test_loader:
                    data = torch.tensor(data).to(torch.float32).to('cuda')

                    output = model(data)

                    data_grid = data[:, :size[0][0], :size[0][1]].squeeze().detach().cpu() - 1
                    output_grid = torch.round(output[:, :size[0][0], :size[0][1]]).squeeze().detach().cpu() - 1

                    # output = output[:,:size[0][0], :size[0][1]].detach().cpu()
                    round_output = torch.round(output[:, :size[0][0], :size[0][1]]) - 1

                    if use_batch or not use_size:
                        loss = criteria(output, data)
                    else:
                        loss = criteria(output[:, :, :size[0][0] * size[0][1]],
                                        data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))

                    if torch.equal(round_output, data[:, :size[0][0], :size[0][1]] - 1):
                        total_acc += 1
                    else:
                        pass
                    # total_loss.append(loss)
                    count += 1
                if use_wandb:
                    wandb.log({
                        "test_loss": sum(train_loss) / len(train_loss),
                        "test_acc": f'acc: {total_acc / count}({total_acc}/{count})'
                    }, step=epoch)
                pbar.write(f'acc: {total_acc / count}({total_acc}/{count})')

        if not use_pretrain and epoch % 50 == 0:
            torch.save(model.state_dict(), 'auto_mse_temp.pt')
    if not use_pretrain:
        if not use_size:
            torch.save(model.state_dict(), f'No_size_auto_mse_{epochs}.pt')
        else:
            torch.save(model.state_dict(), f'auto_mse_{epochs}.pt')

def MSE_method(args, train_loader, valid_loader, test_loader):
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    use_pretrain = args.use_pretrain
    use_valid = args.use_valid
    use_test = args.use_test
    use_batch = args.use_batch
    train_auto = args.train_auto

    model = New_Autoencoder_batch1_v2().to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criteria = nn.MSELoss().to('cuda')
    pbar = tqdm(range(1,epochs+1))
    for epoch in pbar:
        if use_pretrain:
            model.load_state_dict(torch.load('new2_200.pt'))
        else:
            train_loss = []
            for x, y, x_size, y_size in train_loader:
                x = torch.tensor(x).to(torch.float32).to('cuda')
                y = torch.tensor(y).to(torch.float32).to('cuda')
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                output = model(x)
                x_grid = x[:, :int(x_size[0]), :int(x_size[1])].detach().cpu().squeeze() -1
                y_grid = y[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() -1
                output_grid = output[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() -1

                # plt.imshow(y_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
                # plt.imshow(output_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)

                loss = criteria(y[:,:int(y_size[0]), :int(y_size[1])],output[:,:int(y_size[0]), :int(y_size[1])])
                train_loss.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            pbar.write(f'train_loss: {sum(train_loss) / len(train_loss)}')
            # print(f'train_loss: {sum(train_loss) / len(train_loss)}')

        total_acc = 0
        total_loss = []
        count = 0

        if use_valid:
            for x, y, x_size, y_size in valid_loader:
                x = torch.tensor(x).to(torch.float32).to('cuda')
                y = torch.tensor(y).to(torch.float32).to('cuda')
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                y = y[:,:int(y_size[0]), :int(y_size[1])]
                output = model(x)

                x_grid = x[:, :int(x_size[0]), :int(x_size[1])].detach().cpu().squeeze() - 1
                y_grid = y[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1
                output_grid = torch.round(output[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1)

                output = output[:,:int(y_size[0]), :int(y_size[1])]
                round_output = torch.round(output)
                if torch.equal(round_output,y):
                  total_acc += 1

                count += 1

            pbar.write(f'acc: {total_acc / count}({total_acc}/{count})')

        # print(f'{epoch}번째 epoch: acc_{total_acc}, total_loss_{total_acc}')

    torch.save(model.state_dict(), f'new2_{epochs}.pt')

def New_MSE_method(args, train_loader, valid_loader, test_loader):
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    use_pretrain = args.use_pretrain
    use_valid = args.use_valid
    use_test = args.use_test
    use_batch = args.use_batch
    train_auto = args.train_auto

    model = New_Autoencoder_batch1_v2().to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criteria = nn.MSELoss().to('cuda')
    pbar = tqdm(range(1,epochs+1))
    for epoch in pbar:
        if use_pretrain:
            model.load_state_dict(torch.load('new1_200.pt'))
        else:
            train_loss = []
            for x, y, x_size, y_size in train_loader:
                x = torch.tensor(x).to(torch.float32).to('cuda')
                y = torch.tensor(y).to(torch.float32).to('cuda')
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                output = model(x)
                x_grid = x[:, :int(x_size[0]), :int(x_size[1])].detach().cpu().squeeze() -1
                y_grid = y[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() -1
                output_grid = output[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() -1

                # plt.imshow(y_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
                # plt.imshow(output_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)

                loss = criteria(y[:,:int(y_size[0]), :int(y_size[1])],output[:,:int(y_size[0]), :int(y_size[1])])
                train_loss.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            pbar.write(f'train_loss: {sum(train_loss) / len(train_loss)}')
            # print(f'train_loss: {sum(train_loss) / len(train_loss)}')

        total_acc = 0
        total_loss = []

        for x, y, x_size, y_size in valid_loader:
            x = torch.tensor(x).to(torch.float32).to('cuda')
            y = torch.tensor(y).to(torch.float32).to('cuda')
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            y = y[:,:int(y_size[0]), :int(y_size[1])]
            output = model(x)

            x_grid = x[:, :int(x_size[0]), :int(x_size[1])].detach().cpu().squeeze() - 1
            y_grid = y[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1
            output_grid = torch.round(output[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1)

            output = output[:,:int(y_size[0]), :int(y_size[1])]
            round_output = torch.round(output)
            if torch.equal(round_output,y):
              total_acc += 1
            # total_loss.append(loss)

        # pbar.write(f'{epoch}번째 epoch: acc_{total_acc}, total_loss_{total_acc}')

        # print(f'{epoch}번째 epoch: acc_{total_acc}, total_loss_{total_acc}')

    torch.save(model.state_dict(), f'new_{epochs}.pt')


def Auto_Cross_method(args, train_loader, valid_loader, test_loader):
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    use_pretrain = args.use_pretrain
    use_train = args.use_train
    use_valid = args.use_valid
    use_test = args.use_test
    use_batch = args.use_batch
    use_size = args.use_size
    use_wandb = args.use_wandb
    train_auto = args.train_auto
    pretrain = args.pretrain
    model = Autoencoder2().to('cuda')
    # model = Autoencoder_batch1().to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if use_pretrain:
        model.load_state_dict(torch.load(pretrain), strict=False)
    # criteria = nn.NLLLoss().to('cuda')
    pbar = tqdm(range(1, epochs + 1))
    criteria = nn.CrossEntropyLoss().to('cuda')
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
    #                                         lr_lambda=lambda epoch: 0.95 ** epoch,
    #                                         last_epoch=-1,
    #                                         verbose=False)

    # if not use_pretrain:
    for epoch in pbar:
        count = 0
        train_loss = []
        if use_train:
            for data, size in train_loader:

                data = torch.tensor(data).to(torch.float32).to('cuda')
                if use_batch:
                    data = data.unsqueeze(1)
                # y = y[:,:int(x_size[0]), :int(x_size[1])]
                # data = data.reshape(1, -1)

                output = model(data)

                output = output.unsqueeze(-1)
                output = model.proj(output)
                output = output.view(batch_size, -1, 11)
                # output = torch.softmax(output,dim=-1)
                output = output.permute(0, 2, 1)
                view_output = torch.argmax(torch.softmax(output, dim=1), dim=1)

                data_grid = data[:, :size[0][0], :size[0][1]].detach().cpu() - 1
                output_grid = torch.round(output[:, :size[0][0], :size[0][1]].detach().cpu() - 1)

                # plt.imshow(y_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
                # plt.imshow(view_output, cmap=grid_visual.cmap, norm=grid_visual.norm)

                # loss = criteria(output,y)
                if use_batch or not use_size:
                    loss = criteria(output,data.view(batch_size, -1).to(torch.long))
                else:
                    loss = criteria(output[:, :, :size[0][0]*size[0][1]], data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss)
                count += 1
            # scheduler.step()
            if use_wandb:
                wandb.log({
                    "train_loss": sum(train_loss) / len(train_loss),
                }, step=epoch)
            print(f'train_loss: {sum(train_loss)/len(train_loss)}')
        total_acc = 0
        total_loss = []


        if args.use_valid:
            count = 0
            valid_loss = []
            for data, size in valid_loader:
                data = torch.tensor(data).to(torch.float32).to('cuda')
                if use_batch:
                    data = data.unsqueeze(1)

                output = model(data)

                # output = output[:, :int(size[0][0]), :int(size[0][1])]

                # output = output[:,:size[0][0], :size[0][1]].detach().cpu()

                output = output.unsqueeze(-1)
                output = model.proj(output)
                output = output.reshape(batch_size, -1, 11)
                output = output.permute(0, 2, 1)

                if use_batch or not use_size:
                    loss = criteria(output, data.view(batch_size, -1).to(torch.long))
                else:
                    loss = criteria(output[:, :, :size[0][0] * size[0][1]],
                                    data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))

                output = torch.argmax(torch.softmax(output, dim=1), dim=1).reshape(batch_size, 30, 30) #int(size[0][0]), int(size[0][1]))

                # view_output = torch.argmax(torch.softmax(output, dim=1), dim=1)[:, :size[0][0], :size[0][1]]

                data_grid = data.squeeze(1)[:, :size[0][0], :size[0][1]].squeeze().detach().cpu() - 1
                output_grid = torch.round(output[:, :size[0][0], :size[0][1]]).squeeze().detach().cpu() - 1

                # plt.imshow(data_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
                # plt.imshow(view_output.squeeze().detach().cpu(), cmap=grid_visual.cmap, norm=grid_visual.norm)

                view_output = output[:, :size[0][0], :size[0][1]] - 1

                valid_loss.append(loss)
                if torch.equal(view_output, data[:, :size[0][0], :size[0][1]] - 1):
                    total_acc += 1
                else:
                    pass
                # total_loss.append(loss)
                count += 1
            if use_wandb:
                wandb.log({
                    "valid_loss": sum(valid_loss) / len(valid_loss),
                }, step=epoch)

            pbar.write(f'acc: {total_acc / count}({total_acc}/{count})')
            print(sum(valid_loss) / len(valid_loss))

        if use_test:
            count = 0
            test_loss = []
            for data, size in test_loader:
                data = torch.tensor(data).to(torch.float32).to('cuda')
                if use_batch:
                    data = data.unsqueeze(1)
                output = model(data)

                # output = output[:, :int(size[0][0]), :int(size[0][1])]

                # output = output[:,:size[0][0], :size[0][1]].detach().cpu()

                output = output.unsqueeze(-1)
                output = model.proj(output)
                output = output.reshape(batch_size, -1, 11)
                output = output.permute(0, 2, 1)

                if use_batch or not use_size:
                    loss = criteria(output, data.view(batch_size, -1).to(torch.long))
                else:
                    loss = criteria(output[:, :, :size[0][0] * size[0][1]],
                                    data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))

                output = torch.argmax(torch.softmax(output, dim=1), dim=1).reshape(batch_size, 30, 30)# int(size[0][0]), int(size[0][1]))

                # view_output = torch.argmax(torch.softmax(output, dim=1), dim=1)[:, :size[0][0], :size[0][1]]

                data_grid = data[:, :size[0][0], :size[0][1]].squeeze().detach().cpu() - 1
                output_grid = torch.round(output[:, :size[0][0], :size[0][1]]).squeeze().detach().cpu() - 1

                view_output = output[:, :size[0][0], :size[0][1]] - 1

                # plt.imshow(y_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
                # plt.imshow(view_output, cmap=grid_visual.cmap, norm=grid_visual.norm)
                test_loss.append(loss)
                if torch.equal(view_output, data[:, :size[0][0], :size[0][1]] - 1):
                    total_acc += 1
                else:
                    pass
                # total_loss.append(loss)
                count += 1
            if use_wandb:
                wandb.log({
                    "test_loss": sum(test_loss) / len(test_loss),
                }, step=epoch)

            pbar.write(f'acc: {total_acc / count}({total_acc}/{count})')
            print(sum(test_loss) / len(test_loss))

    if not use_pretrain and epoch % 50 == 0:
        torch.save(model.state_dict(), 'auto_cross_temp.pt')
    if not use_pretrain:
        if not use_size:
            torch.save(model.state_dict(), f'No_size_auto_cross_{epochs}.pt')
        else:
            torch.save(model.state_dict(), f'auto_cross_{epochs}.pt')


def train_action_mse():
    model = Autoencoder1_2().to('cuda')
    model.load_state_dict(torch.load('sample1.pt'))
    model.encoder.training = False
    model.decoder.training = False
    model.encoder[0].bias.requires_grad  = False
    model.encoder[0].weight.requires_grad  = False
    model.encoder[2].bias.requires_grad  = False
    model.encoder[2].weight.requires_grad  = False
    model.decoder[0].bias.requires_grad  = False
    model.decoder[0].weight.requires_grad  = False
    model.decoder[2].bias.requires_grad  = False
    model.decoder[2].weight.requires_grad  = False

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criteria = nn.MSELoss().to('cuda')
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train_loss = []
        for x, y, x_size, y_size in train_loader:
            x = torch.tensor(x).to(torch.float32).to('cuda')
            y = torch.tensor(y).to(torch.float32).to('cuda')
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            output = model(x)
            x_grid = x[:, :int(x_size[0]), :int(x_size[1])].detach().cpu().squeeze() - 1
            y_grid = y[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1
            output_grid = output[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1

            # plt.imshow(y_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
            # plt.imshow(output_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)

            loss = criteria(y[:, :int(y_size[0]), :int(y_size[1])], output[:, :int(y_size[0]), :int(y_size[1])])
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pbar.write(f'train_loss: {sum(train_loss) / len(train_loss)}')
        # print(f'train_loss: {sum(train_loss) / len(train_loss)}')

        total_acc = 0
        total_loss = []

        for x, y, x_size, y_size in valid_loader:
            x = torch.tensor(x).to(torch.float32).to('cuda')
            y = torch.tensor(y).to(torch.float32).to('cuda')
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            y = y[:, :int(y_size[0]), :int(y_size[1])]
            output = model(x)

            x_grid = x[:, :int(x_size[0]), :int(x_size[1])].detach().cpu().squeeze() - 1
            y_grid = y[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1
            output_grid = output[:, :int(y_size[0]), :int(y_size[1])].detach().cpu().squeeze() - 1

            output = output[:, :int(y_size[0]), :int(y_size[1])]
            round_output = torch.round(output)
            if torch.equal(round_output, y):
                total_acc += 1
            total_loss.append(loss)

        # pbar.write(f'{epoch}번째 epoch: acc_{total_acc}, total_loss_{total_acc}')

        # print(f'{epoch}번째 epoch: acc_{total_acc}, total_loss_{total_acc}')

    torch.save(model.state_dict(), 'sample_final_1-2.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrain', type=str, default='auto_200.pt')
    parser.add_argument('--use_permute', action='store_true', default=False)
    parser.add_argument('--use_pretrain', action='store_true', default=False)
    parser.add_argument('--use_train', action='store_true', default=False)
    parser.add_argument('--use_valid', action='store_true', default=False)
    parser.add_argument('--use_test', action='store_true', default=False)
    parser.add_argument('--use_batch', action='store_true', default=False)
    parser.add_argument('--use_size', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--train_auto', action='store_true', default=False)
    parser.add_argument('--kind_of_loss', type=str, default='mse')

    args = parser.parse_args()

    if args.batch_size > 1 and not args.use_batch:
        raise f'batch_size가 1보다 큰데 ues_batch가 false임'

    if args.train_auto:
        train_dataset = ARCDataset('train_auto_data.json', mode='auto', permute_mode=args.use_permute)
        valid_dataset = ARCDataset('valid_auto_data.json', mode='auto', permute_mode=args.use_permute)
        test_dataset = ARCDataset('test_auto_data.json', mode='auto', permute_mode=args.use_permute)
    else:
        train_dataset = ARCDataset('train_data.json')  # , mode='auto')
        valid_dataset = ARCDataset('valid_data.json')  # , mode='auto')
        test_dataset = ARCDataset('test_auto_data.json')  # , mode='auto')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    if args.use_wandb:
        tm = localtime(time.time())
        run = wandb.init(project=f'Auto_encoder_train_auto_{args.train_auto}_loss_{args.kind_of_loss}', entity='whatchang',)
        config = {
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
        }
        wandb.config.update(config)
        wandb.run.name = f'b{args.batch_size}_e{args.epochs}_d{tm.tm_mday}_h{tm.tm_hour}_m{tm.tm_min}_s{tm.tm_sec}'
        wandb.run.save()

    if args.train_auto:
        if args.kind_of_loss == 'mse':
            Auto_MSE_method(args, train_loader, valid_loader, test_loader)
        else:
            Auto_Cross_method(args, train_loader, valid_loader, test_loader)
    else:
        MSE_method(args, train_loader, valid_loader, test_loader)