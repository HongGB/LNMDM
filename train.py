from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config_orig import SAVE_FREQ, LR, WD, save_dir, INPUT_SIZE, resume
from utils import init_log, progress_bar
from dataset import get_main_loaders
import argparse
from torchvision import transforms
import torch as T
import torch.nn as nn
import model_seg
from PIL import Image
import numpy as np
import pickle
import loss
import _init_paths
from config import config
from config import update_config
from glob import glob
from eval_util import AverageMeter

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

log_dir = os.path.join(save_dir, 'log2')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

train_data = pd.read_csv('train_data.csv', header=None)
wsi_name = list(train_data.iloc[:, 0])
wsi_name = wsi_name[1:]
png_name = list(train_data.iloc[:, 1])
png_name = png_name[1:]

target_path = 'img_path'
label_path = 'mask_path'
train_lst = []
train_label = []
for per_wsi,per_png in zip(wsi_name, png_name):
    train_lst.append(target_path + per_wsi + '/' + per_png)
    train_label.append(label_path + per_wsi + '/' + per_png)

test_data = pd.read_csv('test_data.csv', header=None)
wsi_name = list(test_data.iloc[:, 0])
wsi_name = wsi_name[1:]
png_name = list(test_data.iloc[:, 1])
png_name = png_name[1:]

test_lst = []
test_label = []
for per_wsi,per_png in zip(wsi_name, png_name):
    test_lst.append(target_path + per_wsi + '/' + per_png)
    test_label.append(label_path + per_wsi + '/' + per_png)


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--testModel', help='testModel', type=str, default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

args = parse_args()


args.train_files = train_lst
args.train_labels = train_label
print('train num:', len(args.train_files), len(args.train_labels))

args.test_files = test_lst
args.test_labels = test_label
print('test num:', len(args.test_files), len(args.test_labels))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
args.my_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        normalize
    ])

args.mask_transform = transforms.Compose([
        transforms.Resize(size=INPUT_SIZE, interpolation=Image.NEAREST),
        # transforms.ToTensor()
    ])

train_loader, test_loader = get_main_loaders(args)

# define model
net_cls = model_seg.get_model(config)

if resume:
    ckpt = torch.load(resume)
    net_cls.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    train_seg_loss = ckpt['train_seg_loss']
    print(start_epoch)
    print(train_seg_loss)

# define optimizers
raw_parameters = list(net_cls.pretrained_model_seg.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)

schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              ]
# cudnn related setting
cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED


net_cls = net_cls.cuda()
net_cls = DataParallel(net_cls)
writer = SummaryWriter(log_dir)
global_step = 0
train_loss_lst = []
train_step_lst = []
test_loss_lst = []
test_epoch_lst = []
for epoch in range(0, 140):
    # begin training
    _print('--' * 50)
    _print('epoch:{}'.format(epoch))
    net_cls.train()
    step = 0
    for i, data in enumerate(train_loader):
        img, label, info = data[0].cuda(), data[1].cuda(), data[2]

        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        seg_output_higher = net_cls(img)
        # print(seg_output_higher.size())
        
        seg_loss_higher = loss.supervise_segmentloss(seg_output_higher, label)
        train_loss_lst.append(seg_loss_higher.data.item())
        train_step_lst.append(global_step)
        writer.add_scalar('training seg loss', seg_loss_higher, global_step)

        if step % 50 == 0:
            _print(
            'epoch:{} - step:{} - train seg loss: {:.3f}'.format(
                epoch,
                step,
                seg_loss_higher.item()))
        
        seg_loss_higher.backward()

        raw_optimizer.step()
        
        step += 1
        global_step += 1
        progress_bar(i, len(train_loader), 'train')

    for scheduler in schedulers:
        scheduler.step()

    net_cls.eval()
    val_raw_loss = AverageMeter()
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            img, label, info = data[0].cuda(), data[1].cuda(), data[2]
            # print(label.size())
            batch_size = img.size(0)
            seg_output_higher = net_cls(img)
            # print(seg_output_higher.size())
            seg_loss_higher = loss.supervise_segmentloss(seg_output_higher, label)
            val_raw_loss.update(seg_loss_higher.data.item(), batch_size)

    v_raw_loss = val_raw_loss.avg
    writer.add_scalar('val seg loss', v_raw_loss, epoch)
    test_loss_lst.append(v_raw_loss)
    test_epoch_lst.append(epoch)

    net_state_dict = net_cls.module.state_dict()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save({
                 'epoch': epoch,
                 'train_seg_loss': seg_loss_higher,
                 'net_state_dict': net_state_dict},
                 os.path.join(save_dir, '%03d.ckpt' % epoch))
    result = pd.DataFrame()
    result_test = pd.DataFrame()
    result['train_loss'] = train_loss_lst
    result['step'] = train_step_lst
    result_test['test_loss'] = test_loss_lst
    result_test['epoch'] = test_epoch_lst
    result.to_csv(os.path.join('bladder_training_record.csv'), index=None)
    result_test.to_csv(os.path.join('bladder_val_record.csv'), index=None)

print('finishing training')
