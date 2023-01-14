# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    pred = pred.numpy()
    seg_pred = np.asarray(pred, dtype=np.uint8)
    seg_pred = seg_pred.reshape(-1, 1)

    label = label.numpy()
    seg_gt = np.ones((label.shape[1], label.shape[2]), dtype=np.int)
    cn_index = label[0, :, :] == 1
    seg_gt[cn_index] = 0
    seg_gt = seg_gt.reshape(-1, 1)

    cn_gt = np.where(seg_gt == 0)[0]
    cn_gt_number = len(cn_gt)

    cn_pred = np.where(seg_pred == 0)[0]
    cn_pred_number = len(cn_pred)

    tp = list(set(cn_gt).intersection(set(cn_pred)))
    tp_number = len(tp)
    iou_per = tp_number / float(cn_gt_number + cn_pred_number - tp_number)
    print('iou_per: %.3f' % iou_per)

    fp = list(set(cn_pred).difference(set(cn_gt)))
    fp_number = len(fp)
    fn = list(set(cn_gt).difference(set(cn_pred)))
    fn_number = len(fn)
    dice_per = (2 * tp_number) / float(2*tp_number + fp_number + fn_number)
    print('dice_per: %.3f' % dice_per)

    #seg_gt = np.expand_dims(seg_gt, 0)
    #ignore_index = seg_gt != ignore
    #seg_gt = seg_gt[ignore_index]
    #seg_pred = seg_pred[ignore_index]

    #cm = confusion_matrix(seg_gt, seg_pred, labels=[0, 1])

    '''
    index = (seg_gt * num_class + seg_pred).astype('int32')
    index = np.squeeze(index, 0)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    '''
    return iou_per, dice_per

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr