from torch import nn
import torch
import os
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw
import torchvision.utils as vutils
from torchvision import transforms
import random
import math
import logging
import seg_hrnet
from bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class seg_net(nn.Module):
    def __init__(self, config):
        super(seg_net, self).__init__()
        
        self.pretrained_model_seg = seg_hrnet.get_seg_model(config)

    def forward(self, x_img):
        
        seg_output_hrnet = self.pretrained_model_seg(x_img)
        
        return seg_output_hrnet
    
    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            # pretrained_dict = torch.load(pretrained)
            ckpt = torch.load(pretrained)
            pretrained_dict = ckpt['net_state_dict']
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_model(config):
    model = seg_net(config)
    return model
