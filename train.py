#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import torch
import torch.nn as nn
import time
import argparse
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import visdom
from torch.autograd import Variable
from tqdm import tqdm
from collections import OrderedDict
import shutil

from utils.make_dataset import OSCD_TRAIN,OSCD_TEST  
from networks.net import TransCDNet
from networks import configs as cfg

import utils.evaluate as eva
from utils.loss import l1_loss,DiceLoss
from utils import utils 
from torch.nn.modules.loss import CrossEntropyLoss


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_network(ncfg, tcfg, vis, net_name):
     
    OUTPUTS_DIR = tcfg.path['outputs']
    WEIGHTS_SAVE_DIR = tcfg.path['weights_save_dir']
    BEST_WEIGHTS_SAVE_DIR = tcfg.path['best_weights_save_dir']
    utils.save_cfg(ncfg, tcfg, OUTPUTS_DIR )
    
    init_epoch = 0
    best_f1 = 0
    total_steps = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    train_data = OSCD_TRAIN(tcfg.path['train_txt'], tcfg.path['dataset'], tcfg.im_size, tcfg.dataset_name)
    train_dataloader = DataLoader(train_data, batch_size=tcfg.batch_size, shuffle=True)
    
    
    
    net = TransCDNet(ncfg, tcfg.im_size, False).to(device=device)


    if tcfg.resume:
        assert os.path.exists(os.path.join(WEIGHTS_SAVE_DIR, 'current_net.pth')), 'There is not found any saved weights'
        print("\nLoading pre-trained networks.")
        init_epoch = torch.load(os.path.join(WEIGHTS_SAVE_DIR, 'current_net.pth'))['epoch']
        net.load_state_dict(torch.load(os.path.join(WEIGHTS_SAVE_DIR, 'current_net.pth'))['model_state_dict'])
        with open(os.path.join(OUTPUTS_DIR, 'val_f1.txt')) as f:
            lines = f.readlines()
            best_f1 = float(lines[-1].strip().split(':')[-1])
        print("\tDone.\n")
        

    l_con = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=tcfg.optimizer['lr'], betas=(0.5, 0.999))
    
    start_time = time.time()
    for epoch in range(init_epoch+1, tcfg.epoch+1):
        loss = []
        net.train()
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):

            x1, x2, gt = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            gt = gt.to(device, dtype=torch.float)
             
            epoch_iter += tcfg.batch_size
            total_steps += tcfg.batch_size
             
            #forward
            out = net(x1, x2)
            err = l_con(out, gt)
        
            #backward
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
             
            errors = utils.get_errors(err)            
            loss.append(err.item())
             
            counter_ratio = float(epoch_iter) / len(train_dataloader.dataset)
            if(i % tcfg.print_rate == 0 and i > 0):
                print('Time:{},epoch:{},iteration:{},loss:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i, np.mean(loss)))
                with open(os.path.join(OUTPUTS_DIR,'train_loss.txt'),'a') as f:
                    f.write('Time:{},epoch:{}, iteration:{}, loss:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i, np.mean(loss)))
                    f.write('\n')      
                if tcfg.display:
                    utils.plot_current_errors(epoch, counter_ratio, errors, vis)
                    utils.display_current_images(gt.data, out.data, vis)
        if not os.path.exists(WEIGHTS_SAVE_DIR):
            os.makedirs(WEIGHTS_SAVE_DIR)
        utils.save_weights(epoch,net,optimizer,WEIGHTS_SAVE_DIR, 'net')
        duration = time.time()-start_time
        print('training duration is %g'%duration)



        #val phase
        print('Validating.................')
        pretrained_dict = torch.load(os.path.join(WEIGHTS_SAVE_DIR,'current_net.pth'))['model_state_dict']
        device = torch.device('cuda')
        net_v = TransCDNet(ncfg, tcfg.im_size, False).to(device=device)
        net_v.load_state_dict(pretrained_dict,False)
        with net_v.eval() and torch.no_grad(): 
            out = [[] for _ in range(5)]
            for category in tcfg.category:
                val_dir = os.path.join(tcfg.path['txt'], category + '.txt')
                val_data = OSCD_TEST(tcfg.path['val_txt'], tcfg.path['dataset'], tcfg.im_size, tcfg.dataset_name)
                val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
                TP = 0
                FN = 0
                FP = 0
                TN = 0       
                for k, data in enumerate(val_dataloader):
                    x1, x2, label = data
                    x1 = x1.to(device, dtype=torch.float)
                    x2 = x2.to(device, dtype=torch.float)
                    label = label.to(device, dtype=torch.float)
                    time_i = time.time()
                    v_out = net_v(x1, x2)
                      
                    tp, fp, tn, fn = eva.confuse_matrix(v_out, label)    
                    TP += tp
                    FN += fn
                    TN += tn
                    FP += fp
                metrics = eva.eva_metrics(TP, FP, TN, FN)  
                print('Category:{}, f1:{}'.format(category, metrics[3]))
                for i in range(5):
                    out[i].append(metrics[i])
        
        f1 = np.mean(out[3])
        if not os.path.exists(BEST_WEIGHTS_SAVE_DIR):
            os.makedirs(BEST_WEIGHTS_SAVE_DIR)
        if f1 > best_f1: 
            best_f1 = f1
            shutil.copy(os.path.join(WEIGHTS_SAVE_DIR,'current_net.pth'),os.path.join(BEST_WEIGHTS_SAVE_DIR, net_name+'.pth'))           
        with open(os.path.join(OUTPUTS_DIR,'val_f1.txt'),'a') as f:
            f.write('Time:{},current_epoch:{},current_f1:{},best_f1:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), epoch, f1, best_f1))
            f.write('\n')   
        with open(os.path.join(OUTPUTS_DIR,'val_performance.txt'),'a') as f:
            f.write('Time:{},current_epoch:{},f1:{},precision:{},recall:{},oa:{},kappa:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
                    epoch, np.mean(out[3]), np.mean(out[0]), np.mean(out[2]), np.mean(out[1]), np.mean(out[4])))
            f.write('\n')  
        
        print('Overall evaluation:  current f1 {}, best f1 {}'.format(f1, best_f1))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--net_cfg', type=str, default='SViT_E1_D1_16',help='choice a TransCD model')
    parser.add_argument(
        '--train_cfg', type=str, default='CDNet_2014',help='CDNet_2014 or VL_CMU_CD')
    opt = parser.parse_args()
    
    net_cfg = cfg.CONFIGS[opt.net_cfg]
    train_cfg = cfg.CONFIGS[opt.train_cfg]
    name = opt.net_cfg
    
    if train_cfg.display:
        vis = visdom.Visdom(server="http://localhost", port=8097)
    else:
        vis = None
    
    train_network(net_cfg, train_cfg, vis, name)
    
    