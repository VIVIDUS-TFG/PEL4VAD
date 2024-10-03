from infer import infer_func
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from utils import setup_seed

from model import XModel
from dataset import *

from train import train_func
from test import test_func
#from infer import infer_func
import argparse
import copy

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_checkpoint(model, ckpt_path):
    if os.path.isfile(ckpt_path):
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
def train(model, train_loader, test_loader, gt, logger):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    logger.info('Model:{}\n'.format(model))
    logger.info('Optimizer:{}\n'.format(optimizer))

    initial_auc, n_far = test_func(test_loader, model, gt, cfg.dataset)
    logger.info('Random initialize {}:{:.4f} FAR:{:.5f}'.format(cfg.metrics, initial_auc, n_far))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    auc_far = 0.0

    st = time.time()
    for epoch in range(cfg.max_epoch):
        loss1, loss2 = train_func(train_loader, model, optimizer, criterion, criterion2, cfg.lamda)
        scheduler.step()

        auc, far = test_func(test_loader, model, gt, cfg.dataset)
        if auc >= best_auc:
            best_auc = auc
            auc_far = far
            best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('[Epoch:{}/{}]: loss1:{:.4f} loss2:{:.4f} | AUC:{:.4f} FAR:{:.5f}'.format(
            epoch + 1, cfg.max_epoch, loss1, loss2, auc, far))

    time_elapsed = time.time() - st
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
    logger.info('Training completes in {:.0f}m {:.0f}s | best {}:{:.4f} FAR:{:.5f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_far))


def main(cfg):
    setup_seed(cfg.seed)

    if cfg.dataset == 'ucf-crime':
        train_data = UCFDataset(cfg, test_mode=False)
        test_data = UCFDataset(cfg, test_mode=True)
    elif cfg.dataset == 'xd-violence':
        train_data = XDataset(cfg, test_mode=False)
        test_data = XDataset(cfg, test_mode=True)
    elif cfg.dataset == 'shanghaiTech':
        train_data = SHDataset(cfg, test_mode=False)
        test_data = SHDataset(cfg, test_mode=True)
    else:
        raise RuntimeError("Do not support this dataset!")

    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True)

    model = XModel(cfg)
    gt = np.load(cfg.gt)
    device = torch.device("cuda")
    model = model.to(device)

    param = sum(p.numel() for p in model.parameters())

    if args.mode == 'train':
        train(model, train_loader, test_loader, gt)

    elif args.mode == 'infer':
        if cfg.ckpt_path is not None:
            load_checkpoint(model, cfg.ckpt_path)
        infer_func(model, test_loader, gt, cfg)

    else:
        raise RuntimeError('Invalid status!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--dataset', default='xd', help='anomaly video dataset')
    parser.add_argument('--mode', default='infer', help='model status: (train or infer)')
    args = parser.parse_args()
    cfg = build_config(args.dataset)
    main(cfg)
