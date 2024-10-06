
import time
from utils import fixed_smooth, slide_smooth
from test import *
import csv
import argparse
from configs import build_config
from utils import setup_seed
from model import XModel
import os
from torch.utils.data import DataLoader
from dataset import *

def infer_func(model, dataloader, gt, cfg, args):
    st = time.time()
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = torch.zeros(0).to(device)
        normal_preds = torch.zeros(0).to(device)
        normal_labels = torch.zeros(0).to(device)
        gt_tmp = torch.tensor(gt.copy()).to(device)

        for i, (v_input, name) in enumerate(dataloader):
            v_input = v_input.float()
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input.to(device)
            seq_len = seq_len.to(device)
            logits, _ = model(v_input, seq_len)
            logits = torch.mean(logits, 0)
            logits = logits.squeeze(dim=-1)

            seq = len(logits)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]

            pred = torch.cat((pred, logits))
            labels = gt_tmp[: seq_len[0]*16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16:]

        pred = list(pred.cpu().detach().numpy())
        pred_binary = [1 if pred_value > 0.35 else 0 for pred_value in pred]
        return pred_binary

def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec

def load_checkpoint(model, ckpt_path):
    if os.path.isfile(ckpt_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_dict = torch.load(ckpt_path, map_location=device)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)

def save_results(results, filename):
    np.save(filename, results)
    
def main(cfg,args):
    setup_seed(cfg.seed)

    test_data = XDataset(cfg, test_mode=True)

    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True)

    model = XModel(cfg)
    gt = np.load(cfg.gt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if cfg.ckpt_path is not None:
        load_checkpoint(model, cfg.ckpt_path)


    results = infer_func(model, test_loader, gt, cfg, args)
    save_results(results, os.path.join(args.output_path, 'results.npy'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--output-path', help='output path')
    parser.add_argument('--rgb-list', help='rgb list path')
    parser.add_argument('--dataset', default='xd', help='anomaly video dataset')
    parser.add_argument('--mode', default='infer', help='model status: (train or infer)')
    parser.add_argument('--evaluate', default='false', help='to infer a video or evaluate model metrics: (false or true)')
    args = parser.parse_args()
    cfg = build_config(args.dataset, args.rgb_list)
    main(cfg,args)