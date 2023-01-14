import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def supervise_segmentloss(seg_output, label):
    criterion = torch.nn.BCEWithLogitsLoss()
    number_class = seg_output.size(1)

    seg_output = torch.nn.functional.interpolate(seg_output, size=(label.size(1), label.size(2)), mode='bilinear',
                                                 align_corners=True)
    # print(seg_output.size())

    seg_output = seg_output.permute(0, 2, 3, 1).contiguous().view((-1, number_class))
    label = label.view(-1, 1)

    positions = (label != 2)
    # print(label[positions].size())
    loss = criterion(seg_output[:, 0][positions[:, 0]], label[positions].float())
    return loss