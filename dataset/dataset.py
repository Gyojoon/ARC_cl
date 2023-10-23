from torch.utils.data import Dataset
import numpy as np
import json
import torch
from collections import OrderedDict
from torchvision import transforms
from PIL import Image

class ARCDataset(Dataset):
    def __init__(self, file_name, mode=None, permute_mode=False, augment=True):
        self.dataset = None
        self.mode = mode
        self.permute_mode = permute_mode
        self.augment = augment
        # self.task_dict = {'SameDifferent': 0, 'Copy': 1, 'MoveToBoundary': 2, 'ExtendToBoundary': 3, 'AboveBelow': 4, 'TopBottom3D': 5, 'CleanUp': 6, 'Order': 7, 'HorizontalVertical': 8, 'TopBottom2D': 9, 'CompleteShape': 10, 'FilledNotFilled': 11, 'ExtractObjects': 12, 'Center': 13, 'Count': 14, 'InsideOutside': 15}
        # self.task_dict의 값을 고정시키는게 좋아 보일듯 -> 항상 key에 해당하는 정수 값들이 항상 바뀜
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            # transforms.RandAugment(2,7),
            transforms.ToTensor()
        ])

        with open(file_name, 'r') as f:
            self.dataset = json.load(f)

        # Create a task dictionary to map tasks to unique integers
        if "task" in self.dataset:
            self.task_dict = {task: idx for idx, task in enumerate(set(self.dataset['task']))}
        else:
            self.task_dict = {}

    def __len__(self):
        if self.mode == 'Auto_encoder':
            return len(self.dataset['data'])
        else:
            return len(self.dataset['input'])

    def __getitem__(self, idx):
        if self.mode == 'Auto_encoder':
            x = self.dataset['data'][idx]
            size = self.dataset['size'][idx]
            if self.permute_mode:
                self.permute_color = np.random.choice(11, 11, replace=False)
                for i in range(30):
                    for j in range(30):
                        x[i][j] = self.permute_color[x[i][j]]

            # Apply augmentations if augment flag is True
            if self.augment:
                x = Image.fromarray((x * 255).astype(np.uint8)) # Assuming x values are between 0 and 1
                x = self.transforms(x)

            return x, torch.tensor(size)
        else:
            x = torch.tensor(self.dataset['input'][idx])
            y = torch.tensor(self.dataset['output'][idx])
            x_size = self.dataset['input_size'][idx]
            y_size = self.dataset['output_size'][idx]

            if self.permute_mode:
                self.permute_color = np.random.choice(11, 11, replace=False)
                for i in range(30):
                    for j in range(30):
                        x[i][j] = self.permute_color[x[i][j]]

            # Apply augmentations if augment flag is True
            if self.augment:
                x_numpy = x.numpy()  # Tensor를 Numpy 배열로 변환
                x_image = Image.fromarray((x_numpy * 255).astype(np.uint8))  # Numpy 배열을 이미지로 변환
                x_image = self.transforms(x_image)
                y_numpy = y.numpy()  # Tensor를 Numpy 배열로 변환
                y_image = Image.fromarray((y_numpy * 255).astype(np.uint8))  # Numpy 배열을 이미지로 변환
                y_image = self.transforms(y_image)

            # Get the task's integer representation using the task dictionary
            task_list = self.dataset.get('task', [])
            if 0 <= idx < len(task_list):
                task_value = task_list[idx]
            else:
                task_value = None

            task = self.task_dict.get(task_value, -1)
                    
            return x, y, x_size, y_size, task

class ARC_ValidDataset(Dataset):
    def __init__(self, file_name, mode=None, permute_mode=False, augment=True):
        self.dataset = None
        self.mode = mode
        self.permute_mode = permute_mode
        self.augment = augment
        # self.task_dict = {'SameDifferent': 0, 'Copy': 1, 'MoveToBoundary': 2, 'ExtendToBoundary': 3, 'AboveBelow': 4, 'TopBottom3D': 5, 'CleanUp': 6, 'Order': 7, 'HorizontalVertical': 8, 'TopBottom2D': 9, 'CompleteShape': 10, 'FilledNotFilled': 11, 'ExtractObjects': 12, 'Center': 13, 'Count': 14, 'InsideOutside': 15}
        # self.task_dict의 값을 고정시키는게 좋아 보일듯 -> 항상 key에 해당하는 정수 값들이 항상 바뀜
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        with open(file_name, 'r') as f:
            self.dataset = json.load(f)

        # Create a task dictionary to map tasks to unique integers
        if "task" in self.dataset:
            self.task_dict = {task: idx for idx, task in enumerate(set(self.dataset['task']))}
        else:
            self.task_dict = {}

    def __len__(self):
        if self.mode == 'Auto_encoder':
            return len(self.dataset['data'])
        else:
            return len(self.dataset['input'])

    def __getitem__(self, idx):
        if self.mode == 'Auto_encoder':
            x = self.dataset['data'][idx]
            size = self.dataset['size'][idx]
            if self.permute_mode:
                self.permute_color = np.random.choice(11, 11, replace=False)
                for i in range(30):
                    for j in range(30):
                        x[i][j] = self.permute_color[x[i][j]]

            # Apply augmentations if augment flag is True
            if self.augment:
                x = Image.fromarray((x * 255).astype(np.uint8)) # Assuming x values are between 0 and 1
                x = self.transforms(x)

            return x, torch.tensor(size)
        else:
            x = torch.tensor(self.dataset['input'][idx])
            y = torch.tensor(self.dataset['output'][idx])
            x_size = self.dataset['input_size'][idx]
            y_size = self.dataset['output_size'][idx]

            # Apply augmentations if augment flag is True
            if self.augment:
                x_numpy = x.numpy()  # Tensor를 Numpy 배열로 변환
                x_image = Image.fromarray((x_numpy * 255).astype(np.uint8))  # Numpy 배열을 이미지로 변환
                x_image = self.transforms(x_image)
                y_numpy = y.numpy()  # Tensor를 Numpy 배열로 변환
                y_image = Image.fromarray((y_numpy * 255).astype(np.uint8))  # Numpy 배열을 이미지로 변환
                y_image = self.transforms(y_image)

            # Get the task's integer representation using the task dictionary
            task_list = self.dataset.get('task', [])
            if 0 <= idx < len(task_list):
                task_value = task_list[idx]
            else:
                task_value = None

            task = self.task_dict.get(task_value, -1)
                    
            return x, y, x_size, y_size, task


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

class New_ARCDataset(Dataset):
  def __init__(self, file_name, mode=None, permute_mode=False):
    self.dataset = None
    self.mode = mode
    self.count_boundary = 2500
    self.count = 0
    self.permute_mode = permute_mode
    self.use_permute_mode = True
    self.permute_color = np.random.choice(11, 11, replace=False)
    with open(file_name, 'r') as f:
        self.dataset = json.load(f)
    if self.mode == 'task':
        task_list = list(set(self.dataset['task']))
        self.task_dict = OrderedDict()
        for i, task in enumerate(task_list):
            self.task_dict[task] = i
    elif 'multi' in mode:
        self.categories = ['Move', 'Color', 'Object', 'Pattern', 'Count', 'Crop', 'Boundary', 'Center', 'Resize', 'Inside', 'Outside', 'Remove', 'Copy', 'Position', 'Direction', 'Bitwise', 'Connect', 'Order', 'Combine', 'Fill']

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
            self.permute_color = np.random.choice(11, 11, replace=False)
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
        return torch.tensor(x), torch.tensor(size)
    elif 'multi-bc' in self.mode:
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]
        multi_labels = []
        if self.use_permute_mode and self.permute_mode and self.count % self.count_boundary == 0:
            if self.count_boundary > 1:
                self.count_boundary -= 1
            # else:
            #     self.use_permute_mode = False
            self.permute_color = np.random.choice(11, 11, replace=False)
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
                    y[i][j] = self.permute_color[y[i][j]]
        for category in self.categories:
            temp = [1 if category in self.dataset['task'][idx] else 0]
            multi_labels.append(temp)
        self.count += 1
        return torch.tensor(x), torch.tensor(y), torch.tensor(multi_labels)
    elif 'multi-soft' in self.mode:
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]
        if self.permute_mode:
            self.permute_color = np.random.choice(11, 11, replace=False)
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
                    y[i][j] = self.permute_color[y[i][j]]
        multi_labels = [1 if category in self.dataset['task'][idx] else 0 for category in self.categories]
        return torch.tensor(x), torch.tensor(y), torch.tensor(multi_labels)
    else:
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]
        if self.permute_mode:
            self.permute_color = np.random.choice(11, 11, replace=False)
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
                    y[i][j] = self.permute_color[y[i][j]]
        task = self.task_dict[self.dataset['task'][idx]]
        return torch.tensor(x), torch.tensor(y), task