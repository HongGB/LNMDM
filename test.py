from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.nn import DataParallel
import torch.nn.parallel
import torch
from datetime import datetime
from config_test import resume, save_dir, TARGET_SIZE
from utils import init_log, progress_bar
from dataset_test import get_main_loaders
import argparse
from torchvision import transforms
import model_test
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pickle
import torchvision
import _init_paths
from config import config
from config import update_config

start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

test_data = 'test_data.pkl'
test_label = 'test_label.pkl'
root = os.path.join('result', 'seg_mask')
root2 = os.path.join('result', 'seg_label')

if not os.path.exists(root):
    os.makedirs(root)
if not os.path.exists(root2):
    os.makedirs(root2)

def parse_args():
    parser = argparse.ArgumentParser(description='Test classification network')
    parser.add_argument('--cfg', help='experiment configure file name', type=str, default='')
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--testModel', help='testModel', type=str, default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

args = parse_args()

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

args.test_files = load_pkl(test_data)
args.test_labels = load_pkl(test_label)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
args.my_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

net_cls = model_test.get_model(config)

if resume:
    ckpt = torch.load(resume)
    net_cls.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    train_seg_loss = ckpt['train_seg_loss']
    print(start_epoch)
    print(train_seg_loss)

net_cls = net_cls.cuda()

net_cls = DataParallel(net_cls)

best_acc = 0

def generate_colors(num_colors):
    
    colors_np = np.array([[0,255,0],[255,0,0]])
   
    return colors_np

def plot_assignment(input_img, assign_hard, num_parts):
    
    # generate the numpy array for colors
    colors = generate_colors(num_parts)
    
    coeff = 0.40

    input_np = np.array(input_img).astype(float)
    assign_hard = np.array(assign_hard).astype(int)

    out_img_array = np.zeros((assign_hard.shape[0], assign_hard.shape[1], 3))
    for i in range(num_parts):
        out_img_array[assign_hard == i] = colors[i]

    input_np = (1 - coeff) * input_np + coeff * out_img_array

    return input_np

def merge_and_save(w, h, target_img_size, output_im_list, crop_name_list, filename, save_dir, save_name, flag):
    if w > target_img_size and h > target_img_size:
        w_mod = w % target_img_size
        h_mod = h % target_img_size
        merge_png = Image.new(flag, (w, h))
        print(len(output_im_list))
        print(len(crop_name_list))

        for crop_img, name in zip(output_im_list, crop_name_list):
            patch_name = name[0].split('_')
            hang = int(patch_name[0])
            lie = int(patch_name[1])

            if ((lie + 1) * target_img_size <= h) and ((hang + 1) * target_img_size <= w):
                x_start = 0
                y_start = 0
                merge_png_start_x = hang * target_img_size
                merge_png_start_y = lie * target_img_size
            elif ((lie + 1) * target_img_size > h) and ((hang + 1) * target_img_size <= w):
                x_start = 0
                y_start = target_img_size - h_mod
                merge_png_start_x = hang * target_img_size
                merge_png_start_y = lie * target_img_size

            elif ((lie + 1) * target_img_size <= h) and ((hang + 1) * target_img_size > w):
                x_start = target_img_size - w_mod
                y_start = 0
                merge_png_start_x = hang * target_img_size
                merge_png_start_y = lie * target_img_size

            elif ((lie + 1) * target_img_size > h) and ((hang + 1) * target_img_size > w):
                x_start = target_img_size - w_mod
                y_start = target_img_size - h_mod
                merge_png_start_x = hang * target_img_size
                merge_png_start_y = lie * target_img_size

            merge_png.paste(crop_img.crop((x_start, y_start, target_img_size, target_img_size)),
                            (merge_png_start_x, merge_png_start_y))

        merge_png.save(os.path.join(save_dir, filename[0].split('/')[-1][0:-4] + save_name))
    else:
        save_img = output_im_list[0]
        save_img.save(os.path.join(save_dir, filename[0].split('/')[-1][0:-4] + save_name))


# begin testing
_print('--' * 50)
_print('epoch:{}'.format(start_epoch))
target_img_size = TARGET_SIZE
net_cls.eval()
test_scale = 1
args.test_scale = test_scale
for i in range(0, len(args.test_files)):
    args.single_file = args.test_files[i]

    if os.path.exists(os.path.join(root2, args.single_file[0].split('/')[-1][0:-4] + '_label.png')):
        print('skipping inferred file ', args.single_file[0])
        continue

    ori_img = Image.open(os.path.join(args.single_file[0])).convert('RGB')
    w = ori_img.size[0]
    h = ori_img.size[1]
    args.single_label = args.test_labels[i]
    test_loader = get_main_loaders(args, test_scale)
    for t, data in enumerate(test_loader):
        with torch.no_grad():
            batch_img, filename, crop_name_list, batch_img_non_normal = data[0], data[1], data[2], data[3]
            batch_size = len(batch_img)
            print(batch_size)
            output_im_list = []
            pred_list = []
            for img, img_non_normal in zip(batch_img, batch_img_non_normal):
                img_non_normal = img_non_normal.squeeze(0)
                img_non_normal = torchvision.transforms.ToPILImage()(img_non_normal)
                seg_output = net_cls(img.cuda())

                if w > target_img_size and h > target_img_size:
                    seg_output_up = torch.nn.functional.interpolate(seg_output.data.cpu(), size=(target_img_size, target_img_size), mode='bilinear', align_corners=True)
                else:
                    seg_output_up = torch.nn.functional.interpolate(seg_output.data.cpu(), size=(h, w), mode='bilinear', align_corners=True)

                seg_output_up = torch.sigmoid(seg_output_up)
                # print(torch.unique(seg_output_up, return_counts=True))
                seg_output_up[seg_output_up > 0.5] = 1
                seg_output_up[seg_output_up < 0.5] = 0
                pred = seg_output_up.squeeze(0)
                # _, pred = torch.max(seg_output_up, 1)
                # colorize and save the assignment

                output_np = plot_assignment(img_non_normal, pred.squeeze(0).numpy(), 2)
                im = Image.fromarray(np.uint8(output_np))
                output_im_list.append(im)

                im_label = Image.fromarray(np.uint8(pred.squeeze(0).numpy()))
                pred_list.append(im_label)


            save_name = '_label.png'
            flag = 'L'
            merge_and_save(w, h, target_img_size, pred_list, crop_name_list, filename, root2, save_name, flag)


            save_name = '_mask.png'
            flag = 'RGB'
            merge_and_save(w, h, target_img_size, output_im_list, crop_name_list, filename, root, save_name, flag)


        progress_bar(i, len(test_loader), 'test')
