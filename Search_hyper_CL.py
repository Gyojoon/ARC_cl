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

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#여러 라이브러리의 난수 생성기의 시드를 동일한 값으로 설정하는 함수
def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def set_wandb(target_name, mode, seed, optimizer, train_batch_size, valid_batch_size, lr, temperature, dataset='ARC', entity='gyojoongu'):
    run = wandb.init(project=f'KSC_CL_Train', entity=entity)
    if mode == 'train':
        config = {
            'optimizer': optimizer,
            'learning_rate': lr,
            'temperature': temperature,
            'epochs': epochs,
            'train_batch_size': train_batch_size,
            'valid_batch_size': valid_batch_size,
            'seed': seed,
        }
        wandb.config.update(config)
        wandb.run.name = f'{model_name}_o{optimizer}_l{lr}_b{batch_size}_e{epochs}_s{seed}'
    wandb.run.save()
    return run

#VAE structure
class vae_Linear_origin(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(11, 512)      #왜 11인지

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

def label_making(task):
    task = task.tolist()
    label_index_list = []
    label_list = []
    for i in range(len(task)):
        if i == 0 or task[i] not in label_list:
            label_index_list.append(task[i])


    label_list = [label_index_list.index(x) for x in task]


    return torch.tensor(label_list, dtype=torch.long)

def nt_xent_loss(output, temperature):
    batch_size = output.shape[0]  
    logits = torch.mm(output, output.t().contiguous()) / temperature
    labels = torch.arange(batch_size).to(output.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

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
        set_wandb('new_idea', 'train', seed, 'Lion', train_batch_size, valid_batch_size, lr, temperature, kind_of_dataset, entity)


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

            loss = nt_xent_loss(output, temperature)

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

            loss = nt_xent_loss(output, temperature)

            valid_total_loss.append(loss) # To store the loss value and not the tensor

        avg_valid_loss = sum(valid_total_loss) / len(valid_total_loss)

        print(f'valid loss: {avg_valid_loss}')

        if use_wandb:
            wandb.log({
                "valid_loss": avg_valid_loss,
            }, step=epoch)
    
    wandb.finish()
    
    return avg_valid_loss

entity = 'whatchang'
train_batch_size = 128
valid_batch_size = 16
lr = 1e-4
batch_size = train_batch_size
epochs = 1000
seed = 777
model_name = 'vae'
mode = 'task'
temperature = 1
use_wandb = True
use_scheduler = False
scheduler_name = 'LROn'
early_stopping = EarlyStopping(patience=100, verbose=True, path='best_model.pt')  # 초기화
#lr_lambda = 0.97

new_model = new_idea_vae('./result/Cross_vae_Linear_origin_b64_lr1e-3_4.pt').to('cuda')         #Cross_vae_Linear_origin_b64_lr1e-3_4.pt이게 뭔지 확인!
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

#optimizer = optim.AdamW(new_model.parameters(), lr=lr)
optimizer = Lion(new_model.parameters(), lr=lr, weight_decay=1e-2)
#optimizer = optim.Adam(new_model.parameters(), lr=lr, weight_decay=5e-4)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: lr_lambda ** epoch)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)


seed_fix(seed)

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
study = optuna.create_study(direction='miniimize', sampler=sampler)
study.optimize(objective, n_trials=100)

# new_model.load_state_dict(torch.load('best_model.pt'))
# torch.save(new_model.state_dict(), f'result/number1.pt')