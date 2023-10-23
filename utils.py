import torch
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np

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

def set_wandb(epochs, mode, seed, optimizer, train_batch_size, valid_batch_size, lr, temperature, dataset='ARC', entity='gyojoongu'):
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
        wandb.run.name = f'o{optimizer}_l{lr}_b{train_batch_size}_e{epochs}_s{seed}'
    wandb.run.save()
    return run

def label_making(task):
    task = task.tolist()
    label_index_list = []
    label_list = []
    for i in range(len(task)):
        if i == 0 or task[i] not in label_list:
            label_index_list.append(task[i])


    label_list = [label_index_list.index(x) for x in task]


    return torch.tensor(label_list, dtype=torch.long)