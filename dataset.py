import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from torchvision import transforms
from config_orig import TARGET_SIZE


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


class Main_Dataset(Dataset):
    def __init__(self, file_list, labels, my_transform, mask_transform, target_image_size=TARGET_SIZE):
        self.file_list = file_list
        self.labels = labels

        self.my_transform = my_transform
        self.mask_transform = mask_transform
        self.target_img_size = target_image_size
        self.num_classes = 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, ind):
        true_idx = ind
        filename = self.file_list[true_idx]
        # print(filename)
        label_png = self.labels[true_idx]
        label_png = label_png[:-4] + '.png'
        Image.MAX_IMAGE_PIXELS = None
        whole_mask = Image.open(os.path.join(label_png))

        img = Image.open(os.path.join(filename)).convert('RGB')
        width, height = img.size
        min_length = min(width, height)
        # print(width, height)

        bb_x1 = width - 1
        bb_y1 = height - 1

        random_target_size = random.randint(self.target_img_size[0], self.target_img_size[1])
        #print(random_target_size)
        if min_length < random_target_size:
            random_target_size = min_length

        #print(random_target_size, min_length)

        if int(bb_x1 - random_target_size) < 0:
            x_min = 0
        else:
            x_min = int(bb_x1 - random_target_size)

        if int(bb_y1 - random_target_size) < 0:
            y_min = 0
        else:
            y_min = int(bb_y1 - random_target_size)

        crop_x0 = random.randint(0, x_min)
        crop_y0 = random.randint(0, y_min)

        if (crop_x0 + random_target_size) > width:
            crop_x1 = width - 1
        else:
            crop_x1 = int(crop_x0 + random_target_size)

        if (crop_y0 + random_target_size) > height:
            crop_y1 = height - 1
        else:
            crop_y1 = int(crop_y0 + random_target_size)

            # print(crop_x0, crop_y0, crop_x1, crop_y1)
        patch_img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        #print(patch_img.size)

        patch_mask_array = whole_mask.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        #print(patch_mask_array.size)

        patch_img = self.my_transform(patch_img)
        patch_mask_array = self.mask_transform(patch_mask_array)
        patch_mask_np = np.array(patch_mask_array)
        patch_mask_tensor = torch.from_numpy(patch_mask_np)

        info = torch.LongTensor([true_idx]).squeeze()
        return patch_img, patch_mask_tensor, info


def get_main_loaders(opt):
    my_transform = opt.my_transform
    train_dataset = Main_Dataset(opt.train_files, opt.train_labels, my_transform, opt.mask_transform,
                                 target_image_size=TARGET_SIZE)
    test_dataset = Main_Dataset(opt.test_files, opt.test_labels, my_transform, opt.mask_transform,
                                 target_image_size=TARGET_SIZE)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=16)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=4)
    return train_loader, test_loader





