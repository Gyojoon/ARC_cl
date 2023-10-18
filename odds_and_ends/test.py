from model import *
from dataset import *
from torch.utils.data import DataLoader
from tester import *
from utils import *
import argparse
import yaml

def main(config):
    # get config infomation
    # config 정보 불러오기
    seed = args.seed

    target_name = config['target_name']
    model_name = config['model_name']
    tester_name = config['tester']

    test_data_name = config['test_data']

    lr = float(config['lr'])
    epochs = config['epochs']
    batch_size = config['batch_size']
    kind_of_loss = config['kind_of_loss'].lower()

    use_wandb = config['use_wandb']
    use_pretrain = config['use_pretrain']

    mode = config['target_name']
    use_permute = config['use_permute']


    # setup data_loader instances
    # dataloader 설정
    test_dataset = ARCDataset(test_data_name, mode=mode, permute_mode=use_permute)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True)

    # setup wandb
    # wandb 설정
    if use_wandb:
        run = set_wandb(target_name, kind_of_loss, lr, 1, batch_size, model_name, use_permute, use_pretrain, mode='test')
    else:
        run = None

    # setup model
    # 모델 설정
    model = globals()[model_name]().to('cuda')

    # setup function handles of loss and metrics
    # loss함수와 metrics 설정
    criterion = set_loss(kind_of_loss).to('cuda')


    tester = globals()[tester_name](model, criterion, config,
                      test_loader, run)

    tester.test_epoch()


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
    main(config)