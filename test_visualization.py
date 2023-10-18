import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

# 그래프 잘그리게 하는 코드
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns

import torch
import torch.nn as nn
from torch import linalg as LA
from mpl_toolkits.mplot3d import Axes3D

from dataset import *
from torch.utils.data import DataLoader

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

class new_idea_vae_cl(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.autoencoder = vae_Linear_origin()
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.first_layer_parameter_size = 128
        self.second_layer_parameter_size = 128

        self.last_parameter_size = 128
        self.num_categories = 16

        self.fusion_layer1 = nn.Linear(2*128*900, self.first_layer_parameter_size)
        self.fusion_layer2 = nn.Linear(self.first_layer_parameter_size, self.second_layer_parameter_size)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.norm_layer1 = nn.BatchNorm1d(self.first_layer_parameter_size)
        self.norm_layer2 = nn.BatchNorm1d(self.second_layer_parameter_size)


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
        fusion_feature = self.leaky_relu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.norm_layer2(fusion_feature)
        output = self.leaky_relu(fusion_feature)

        return output

class new_idea_vae_using_cl(nn.Module):
    def __init__(self, vae_model_file, model_file):
        super().__init__()

        self.vae_cl  = new_idea_vae_cl(vae_model_file)
        self.vae_cl.load_state_dict(torch.load(model_file))
        self.vae_freeze()

        self.first_layer_parameter_size = 128
        self.second_layer_parameter_size = 64
        self.third_layer_parameter_size = 32
        self.last_parameter_size = 128
        self.num_categories = 16


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


        self.AboveBelow_layer = nn.Linear(self.last_parameter_size, 1)
        self.Center_layer = nn.Linear(self.last_parameter_size, 1)
        self.CleanUp_layer = nn.Linear(self.last_parameter_size, 1)
        self.CompleteShape_layer = nn.Linear(self.last_parameter_size, 1)
        self.Copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.Count_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtendToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtractObjects_layer = nn.Linear(self.last_parameter_size, 1)
        self.FilledNotFilled_layer = nn.Linear(self.last_parameter_size, 1)
        self.HorizontalVertical_layer = nn.Linear(self.last_parameter_size, 1)
        self.InsideOutside_layer = nn.Linear(self.last_parameter_size, 1)
        self.MoveToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.Order_layer = nn.Linear(self.last_parameter_size, 1)
        self.SameDifferent_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom2D_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom3D_layer = nn.Linear(self.last_parameter_size, 1)

    def vae_freeze(self):
        for param in self.vae_cl.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        output = self.vae_cl(input_x, output_x)
        return output

class new_idea_vae(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.autoencoder = vae_Linear_origin()
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.first_layer_parameter_size = 128
        self.last_parameter_size = 128

        self.fusion_layer1 = nn.Linear(128*900, self.first_layer_parameter_size)
        self.fusion_layer2 = nn.Linear(self.first_layer_parameter_size, self.last_parameter_size)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.norm_layer1 = nn.BatchNorm1d(self.first_layer_parameter_size)
        self.norm_layer2 = nn.BatchNorm1d(self.last_parameter_size)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]
        if len(input_x.shape) > 3:
            embed_input = self.autoencoder.embedding(input_x.reshape(batch_size, 900).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_input = self.autoencoder.embedding(input_x.reshape(-1, 900).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(-1, 900).to(torch.long))
        input_feature = self.autoencoder.encoder(embed_input)
        output_feature = self.autoencoder.encoder(embed_output)

        # Using difference of the latent vectors
        diff_feature = input_feature - output_feature
        concat_feature = diff_feature.reshape(batch_size, -1)

        fusion_feature = self.fusion_layer1(concat_feature)
        fusion_feature = self.norm_layer1(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature) 

        output = self.fusion_layer2(fusion_feature)
        output = self.norm_layer2(output)
        output = self.leaky_relu(output)

        return output

train_batch_size = 2
valid_batch_size = 1
batch_size = train_batch_size
seed = 777
model_name = 'vae'
mode = 'task'

model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')         #Cross_vae_Linear_origin_b64_lr1e-3_4.pt이게 뭔지 확인!
model.load_state_dict(torch.load('best_model.pt'))
#train_dataset_name = 'data/train_data.json'
#valid_dataset_name = 'data/valid_data.json'
train_dataset_name = 'data/train_new_idea.json'
valid_dataset_name = 'data/valid_new_idea.json'
# train_dataset_name = 'data/train_new_idea_task_sample2_.json'
# valid_dataset_name = 'data/valid_new_idea_task_sample2_.json'
train_dataset = ARCDataset(train_dataset_name, mode=mode)
valid_dataset = ARCDataset(valid_dataset_name, mode=mode)
kind_of_dataset = 'Concept_task_sample2' if 'concept' in train_dataset_name else 'ARC_task_sample2' if 'sample2' in train_dataset_name else 'ARC'
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=True)

n_components = 3

tsne_model = TSNE(n_components=n_components)
pca_model = PCA(n_components=n_components)
data_list = []
target_list = []

for input, output, x_size, y_size, task in train_loader:
    input = input.to(torch.float32).to('cuda')
    output = output.to(torch.float32).to('cuda')
    task = task.to(torch.long).to('cuda')

    output = model(input, output)

    data_list += output.tolist()
    target_list+= task.tolist()

tsne_embedded = tsne_model.fit_transform(torch.tensor(data_list))
pca_embedded = pca_model.fit_transform(torch.tensor(data_list))
#palette = sns.color_palette('bright', 16)

# Concatenate several colormaps
colors1 = plt.cm.viridis(np.linspace(0., 1, 128))
colors2 = plt.cm.plasma(np.linspace(0, 1, 128))
colors3 = plt.cm.inferno(np.linspace(0, 1, 128))
colors4 = plt.cm.magma(np.linspace(0, 1, 36))  # Added a smaller segment from this colormap for diversity

# Stack all the colors into a single array
colors_total = np.vstack((colors1, colors2, colors3, colors4))
mymap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors_total)

sns.scatterplot(x=tsne_embedded[:,0], y=tsne_embedded[:,1], hue=target_list, legend='full', palette=mymap, s = 2)
plt.savefig('number1_tsne2D.png')
sns.scatterplot(x=pca_embedded[:,0], y=pca_embedded[:,1], hue=target_list, legend='full', palette=mymap, s = 2)
plt.savefig('number1_pca2D.png')


# ==================== 경계선 ======================== #

# 3차원 그래프 세팅
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

tsne_df = pd.DataFrame(({
    'x': pd.Series(tsne_embedded[:,0]),
    'y': pd.Series(tsne_embedded[:,1]),
    'z': pd.Series(tsne_embedded[:,2]),
    'target': pd.Series(target_list)
}))


# target 별 분리

tsne_df_class = [0] * 400

for i in range (400):
    tsne_df_class[i] = tsne_df[tsne_df['target'] == i]
    random_color = list(np.random.choice(range(256), size=3) / 255.0)
    ax.scatter(tsne_df_class[i]['x'], tsne_df_class[i]['y'], tsne_df_class[i]['z'], color = random_color, label = i, s = 2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.savefig('number1_tsne3D.png')