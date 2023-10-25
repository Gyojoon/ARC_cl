import torch
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO

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

def set_wandb(epochs, mode, seed, optimizer, train_batch_size, valid_batch_size, lr, temperature, dataset='ARC', entity='gyojoongu', project_name=f'KSC_CL_Train'):
    run = wandb.init(project=project_name, entity=entity)

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
    wandb.run.name = f'{mode}_o{optimizer}_l{lr}_b{train_batch_size}_e{epochs}_s{seed}'
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

def string_to_array(grid):
    # if grid is already in integer form, just return it
    try:
        if isinstance(grid[0][0], int): return grid
    except:
        print(1)

    # mapping = {0:'.',1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h',9:'i',10:'j'}
    # revmap = {v:k for k,v in mapping.items()}
    grid_array = []
    target_grid = grid.replace(']', '').split(', [')

    for index in range(len(target_grid)):
        grid_array.append(np.fromstring(target_grid[index].strip('['), sep=',', dtype=np.int64))

    # newgrid = [[revmap[grid[i][j]] for j in range(len(grid[0]))] for i in range(len(grid))]
    try:
        output_grid_array = np.array(grid_array)
        flag = True
    except:
        output_grid_array = [-1]
        flag = False
    return output_grid_array, flag

def plot_2d_grid(pred_task, label_task, input, output, dataset_dict):
    count = 0
    dataset_dict = list(dataset_dict.keys())
    for pred_, label_, input_, output_ in zip(pred_task, label_task, input, output):

        cvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown",
                  "black"]
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        # fig, axs = plt.subplots(len(data['test']), 3, figsize=(5, len(data['test']) * 3 * 0.7))
        fig, axs = plt.subplots(1, 2, figsize=(5, 1 * 3 * 0.7))
        axs = axs.reshape(-1, 2)  # Reshape axs to have 2 dimensions

        # show grid

        axs[0, 0].set_title(f'Promblem Input {0 + 1}')
        # display gridlines
        rows, cols = np.array(input_).shape
        axs[0, 0].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        axs[0, 0].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        axs[0, 0].grid(True, which='minor', color='black', linewidth=0.5)
        axs[0, 0].set_xticks([]);
        axs[0, 0].set_yticks([])
        axs[0, 0].imshow(np.array(input_), cmap=cmap, vmin=0, vmax=9)

        axs[0, 1].set_title(f'Problem Output {0 + 1}')
        # display gridlines
        rows, cols = np.array(output_).shape
        axs[0, 1].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        axs[0, 1].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        axs[0, 1].grid(True, which='minor', color='black', linewidth=0.5)
        axs[0, 1].set_xticks([]);
        axs[0, 1].set_yticks([])
        axs[0, 1].imshow(np.array(output_), cmap=cmap, vmin=0, vmax=9)
        # plot gpt output if present


        plt.tight_layout()

        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png', dpi=300)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        if count == 0:
            html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        else:
            html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

        html += f'''<p> ====================== Predict task class: {dataset_dict[pred_]} ======================</p>\n'''
        html += f'''<p> ====================== Label task class: {dataset_dict[label_]} ======================</p>\n'''

        # if mode == 'incorrect' and count == 20:
        #     break
        # plt.show()

        # returns back in html format
        count += 1
    return html


def write_file(plot_html, dir_path='result',loss='our', dataset='concept', mode='correct'):
    ''' Writes the output to a html file for easy reference next time '''
    # Create the HTML content
    if mode == 'correct':
        html_content = f'''
        <html>
        <body>
        {plot_html}'''
        html_content += '''
        </body>
        </html>
        '''
    else:
        html_content = f'''
                <html>
                <body>
                {plot_html}'''
        html_content += '''
                </body>
                </html>
                '''

    save_name = f"{dir_path}/{dataset}_{loss}_{mode}.html" if mode == 'correct' else f"{dir_path}/{dataset}_{loss}_{mode}.html"
    # Overwrite if first run
    with open(save_name, 'w') as file:
        file.write(html_content)