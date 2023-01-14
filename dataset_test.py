import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import pickle
import torch
from config_test import TARGET_SIZE


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

class Main_Dataset(Dataset):
    def __init__(self, file_list, my_transform, scale, target_image_size=TARGET_SIZE):
        self.file_list = file_list
        # self.labels = labels

        self.my_transform = my_transform
        self.target_img_size = target_image_size
        self.num_classes = 1
        self.scale = scale

    def __len__(self):
        return 1

    def __getitem__(self, ind):
        filename = self.file_list

        img = Image.open(os.path.join(filename)).convert('RGB')

        # crop the whole testing image
        w = img.size[0]
        h = img.size[1]

        batch_img = []
        crop_name = []
        batch_img_non_normal = []

        if w > self.target_img_size and h > self.target_img_size:
            w_num_patch = w // self.target_img_size
            h_num_patch = h // self.target_img_size
            for k in range(0, (w_num_patch + 1)):
                for j in range(0, h_num_patch + 1):
                    if (k + 1) * self.target_img_size > w and (j + 1) * self.target_img_size < h:
                        crop_patch = img.crop((w - self.target_img_size, j * self.target_img_size, w, (j + 1) * self.target_img_size))
                    elif (k + 1) * self.target_img_size < w and (j + 1) * self.target_img_size > h:
                        crop_patch = img.crop((k * self.target_img_size, (h - self.target_img_size), (k + 1) * self.target_img_size, h))
                    elif (k + 1) * self.target_img_size <= w and (j + 1) * self.target_img_size <= h:
                        crop_patch = img.crop((k * self.target_img_size, j * self.target_img_size, (k + 1) * self.target_img_size, (j + 1) * self.target_img_size))
                    else:
                        crop_patch = img.crop((w - self.target_img_size, h - self.target_img_size, w, h))
                    crop_patch_resize = transforms.Resize((int(self.target_img_size*self.scale), int(self.target_img_size*self.scale)))(crop_patch)
                    crop_patch_t = self.my_transform(crop_patch_resize)
                    print(crop_patch_t.size())
                    batch_img.append(crop_patch_t)
                    crop_name.append(str(k) + '_' + str(j))
                    crop_patch2 = transforms.ToTensor()(crop_patch)
                    batch_img_non_normal.append(crop_patch2)
        else:
            img1 = transforms.Resize((int(h*self.scale), int(w*self.scale)))(img)
            img_t = self.my_transform(img1)
            print(img_t.size())
            batch_img.append(img_t)
            img2 = transforms.ToTensor()(img)
            batch_img_non_normal.append(img2)

        return batch_img, filename, crop_name, batch_img_non_normal


def get_main_loaders(opt):
    test_scale = opt.test_scale
    my_transform = opt.my_transform
    test_dataset = Main_Dataset(opt.single_file, my_transform, test_scale, target_image_size=TARGET_SIZE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2)
    return test_loader





