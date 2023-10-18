from model import *
from dataset import *
from torch.utils.data import DataLoader
from trainer import *
from utils import *
import argparse
import yaml

def main(config, args):
    # get config infomation
    # config 정보 불러오기
    seed = args.seed

    target_name = config['target_name']
    model_name = config['model_name']

    lr = float(config['lr'])
    epochs = config['epochs']
    batch_size = config['batch_size']
    kind_of_loss = config['kind_of_loss'].lower()
    optimizer_name = config['optimizer'].lower()
    scheduler_name = config['scheduler'].lower()
    lr_lambda = config['lr_lambda']
    step_size = config['step_size']
    gamma = config['gamma']

    use_permute = config['use_permute']
    use_wandb = config['use_wandb']

    trainer_name = config['trainer']
    train_dataset_name = config['train_data']
    valid_dataset_name = config['valid_data']

    mode = config['target_name']
    use_permute = config['use_permute']

    # setup data_loader instances
    # dataloader 설정
    train_dataset = ARCDataset(train_dataset_name, mode=mode, permute_mode=use_permute)
    valid_dataset = ARCDataset(valid_dataset_name, mode=mode, permute_mode=use_permute)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, drop_last=True)

    # setup wandb
    # wandb 설정
    if use_wandb:
        run = set_wandb(seed, target_name, kind_of_loss, lr, epochs, batch_size, model_name, use_permute)
    else:
        run = None

    # setup model
    # 모델 설정
    model = globals()[model_name]().to('cuda')

    # setup function handles of loss and metrics
    # loss함수와 metrics 설정
    criterion = set_loss(kind_of_loss).to('cuda')

    # setup optimizer and learning scheduler
    # optimizer와 learning scheduler 설정
    optimizer = set_optimizer(optimizer_name, model, lr)
    scheduler = set_lr_scheduler(optimizer, scheduler_name, lr_lambda, step_size, gamma)

    trainer = globals()[trainer_name](model, criterion, optimizer, config,
                      train_loader, valid_loader, scheduler, run)

    trainer.train_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-s', '--seed', default=777, type=int,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    seed_fix(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config, args)