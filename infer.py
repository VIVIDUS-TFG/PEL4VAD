
import time
from utils import fixed_smooth, slide_smooth
from test import *
import csv
import argparse
from main import main
from configs import build_config
from utils import setup_seed
from log import get_logger
from model import XModel
import os
from torch.utils.data import DataLoader
from dataset import *

def infer_func(model, dataloader, gt, logger, cfg, args):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()

        for i, (v_input, name) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
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

        video_duration = int(np.ceil(len(pred_binary) * 0.96)) # len(pred_binary) = video_duration / 0.96

        if any(pred == 1 for pred in pred_binary):
            message= "El video contiene violencia. "
            message_second = "Los intervalos con violencia son: "
            message_frames = "En un rango de [0-"+ str(len(pred_binary) - 1) +"] los frames con violencia son: "

            start_idx = None
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    if start_idx is None:
                        start_idx = i
                elif start_idx is not None:
                    message_frames += ("[" + str(start_idx) + " - " + str(i - 1) + "]" + ", ") if i-1 != start_idx else ("[" + str(start_idx) + "], ")
                    message_second += ("[" + parse_time(int(np.floor((start_idx + 1)* 0.96))) + " - " + parse_time(int(np.ceil(i * 0.96))) + "], ")
                    start_idx = None

            if start_idx is not None:
                message_frames += ("[" + str(start_idx) + " - " + str(len(pred_binary) - 1) + "]") if len(pred_binary) - 1 != start_idx else ("[" + str(start_idx) + "]")
                message_second += ("[" + parse_time(int(np.floor((start_idx + 1) * 0.96))) + " - " + parse_time(video_duration) + "]")
            else:
                message_frames = message_frames[:-2]              
                message_second = message_second[:-2]              

        else:
            message= "El video no contiene violencia."
            message_frames = ""            
            message_second = ""            

        if args.evaluate == 'true':
            # Create a list of dictionaries to store the data
            data = []
            data.append({
                'video_id': "IDVIDEO",
                'frame_number': pred_binary,
                "violence_label": "1" if any(pred == 1 for pred in pred_binary) else "0",
            })

            # Write the data to a CSV file
            csv_file = 'inference.csv'

            fieldnames = ['video_id', 'frame_number', 'violence_label']
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(data)
        
        time_elapsed = time.time() - st
        print(message + message_frames)
        print(message + message_second)
        print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec

def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
    else:
        logger.info('Not found pretrained checkpoint file.')

def main(cfg,args):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)

    test_data = XDataset(cfg, test_mode=True)

    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True)

    model = XModel(cfg)
    gt = np.load(cfg.gt)
    device = torch.device("cuda")
    model = model.to(device)

    param = sum(p.numel() for p in model.parameters())

    if cfg.ckpt_path is not None:
        load_checkpoint(model, cfg.ckpt_path, logger)
    else:
        logger.info('infer from random initialization')
    infer_func(model, test_loader, gt, logger, cfg, args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--dataset', default='xd', help='anomaly video dataset')
    parser.add_argument('--mode', default='infer', help='model status: (train or infer)')
    parser.add_argument('--evaluate', default='false', help='to infer a video or evaluate model metrics: (false or true)')
    args = parser.parse_args()
    cfg = build_config(args.dataset)
    main(cfg,args)