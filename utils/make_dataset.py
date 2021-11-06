import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torch
from pylab import *
from networks import configs as cfg

TRANSFORM = True

class OSCD_TRAIN(Dataset):
    def __init__(self, txt_path, data_path, im_size, dataset_name):
        super(OSCD_TRAIN, self).__init__()
        self.txt_path = txt_path
        self.data_path = data_path
        self.im_size = im_size
        self.dataset_name = dataset_name
        with open(os.path.join(self.txt_path),'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)
    def __getitem__(self, idx):
        x1 = Image.open(os.path.join(self.data_path, self.list[idx].split(' ')[0]))
        x2 = Image.open(os.path.join(self.data_path, self.list[idx].split(' ')[1]))
        gt = Image.open(os.path.join(self.data_path, self.list[idx].split(' ')[2].strip()))
        
        t = [            
            transforms.RandomRotation((360,360), resample=False, expand=False, center=None),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((180,180), resample=False, expand=False, center=None),
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5 ), (0.5, 0.5, 0.5)),
                   ]
        if TRANSFORM:
            k = np.random.randint(4)        
            x1 = t[k](x1);x2 = t[k](x2);gt = t[k](gt);
            x1 = t[4](x1);x2 = t[4](x2);gt = t[4](gt);
            x1 = t[5](x1);x2 = t[5](x2);
            x1 = t[6](x1);x2 = t[6](x2);
            gt = np.asarray(gt).astype(np.float)
            gt = gt[np.newaxis, :, :]

        if self.dataset_name == 'CDNet_2014':
            return x1, x2, gt
        else:
            return x1, x2, gt/255

    def __len__(self):
        return self.file_size

class OSCD_TEST(Dataset):
    def __init__(self, txt_path, data_path, im_size, dataset_name):
        super(OSCD_TEST, self).__init__()
        self.txt_path = txt_path
        self.data_path = data_path
        self.im_size = im_size
        self.dataset_name = dataset_name
        with open(os.path.join(self.txt_path),'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)
    def __getitem__(self, idx):
        x1 = Image.open(os.path.join(self.data_path, self.list[idx].split(' ')[0]))
        x2 = Image.open(os.path.join(self.data_path, self.list[idx].split(' ')[1]))
        gt = Image.open(os.path.join(self.data_path, self.list[idx].split(' ')[2].strip()))
        
        t = [            
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5 )),
                   ]
        if TRANSFORM:       
            x1 = t[0](x1);x2 = t[0](x2);gt = t[0](gt);
            x1 = t[1](x1);x2 = t[1](x2);
            x1 = t[2](x1);x2 = t[2](x2);
            gt = np.asarray(gt).astype(np.float)
            gt = gt[np.newaxis, :, :]
        if self.dataset_name == 'CDNet_2014':
            return x1, x2, gt
        else:
            return x1, x2, gt/255
    def __len__(self):
        return self.file_size


