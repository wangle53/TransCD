from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import cv2
import argparse
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from torch.nn import functional as F
from networks.net import TransCDNet
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from networks.net import TransCDNet
from networks import configs as cfg
from utils.make_dataset import OSCD_TRAIN,OSCD_TEST
import utils.evaluate as eva
from utils.loss import l1_loss,DiceLoss
from utils import utils 


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_network(category, opt):
    
    ncfg = cfg.CONFIGS[opt.net_cfg]
    tcfg = cfg.CONFIGS[opt.train_cfg]   
    test_dir = os.path.join(tcfg.path['txt'], category + '.txt')
    best_model_path = os.path.join(tcfg.path['best_weights_save_dir'], opt.net_cfg + '.pth')
    pretrained_dict = torch.load(best_model_path)['model_state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = OSCD_TEST(test_dir, tcfg.path['dataset'], tcfg.im_size, tcfg.dataset_name)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    net = TransCDNet(ncfg, tcfg.im_size, False).to(device=device)
    net.load_state_dict(pretrained_dict,False)
    torch.no_grad()
    net.eval()
    i = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i, data in enumerate(test_dataloader):

        x1, x2, gt = data
        x1 = x1.to(device, dtype=torch.float)
        x2 = x2.to(device, dtype=torch.float)
        gt = gt.to(device, dtype=torch.float)

        fake = net(x1, x2)

        save_path = os.path.join(tcfg.path['changemap'], category)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
         
        if opt.save_changemap:
            vutils.save_image(x1.data, os.path.join(save_path,'%d_x1.png'%i), normalize=True)
            vutils.save_image(x2.data, os.path.join(save_path,'%d_x2.png'%i), normalize=True)
            vutils.save_image(fake.data, os.path.join(save_path,'%d_gt_fake.png'%i) , normalize=True)
            vutils.save_image(gt, os.path.join(save_path,'%d_gt.png'%i), normalize=True)
            heatmap = fake.detach().cpu().squeeze()
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path,str(i)+'_heatmap.jpg'),heatmap)
        tp, fp, tn, fn = eva.confuse_matrix(fake, gt)    
        TP += tp
        FN += fn
        TN += tn
        FP += fp
        i += 1
        print(category, 'testing {}th images'.format(i))
    precision, oa, recall, f1, kappa = eva.eva_metrics(TP, FP, TN, FN)
#     print('Catefory:',category)
#     print('Precision is: ',precision)
#     print('Recall is: ',recall)
#     print('OA is: ',oa)
#     print('F1 is: ',f1)
#     print('Kappa is: ',kappa)  
    
    return [precision, oa, recall, f1, kappa], [TP, FN, TN, FP]


def main(opt):
    tcfg = cfg.CONFIGS[opt.train_cfg]  
    categorys = tcfg.category
    out = [[] for _ in range(5)]
    cm = [[] for _ in range(4)]
    for category in categorys:
        out0, out1 = test_network(category, opt)
        for i in range(5):
            out[i].append(out0[i])
        for i in range(4):
            cm[i].append(out1[i])
        with open(os.path.join(tcfg.path['outputs'],'test_score.txt'),'a') as f:
            f.write('Time:{},Category:{},precision:{},oa:{},recall:{},f1:{},kappa:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), 
                                                                                          category, 
                                                                                          out0[0], 
                                                                                          out0[1], 
                                                                                          out0[2],
                                                                                          out0[3],
                                                                                          out0[4],))
            f.write('\n')  
    TP = np.mean(cm[0]); FN = np.mean(cm[1]); TN = np.mean(cm[2]); FP = np.mean(cm[3]);
    pre, oa, re, f1, kappa = eva.eva_metrics(TP, FP, TN, FN)  
    with open(os.path.join(tcfg.path['outputs'],'test_score.txt'),'a') as f:
        f.write('-'*100) 
        f.write('\n') 
        f.write('Time:{},Macro eva,precision:{},oa:{},recall:{},f1:{},kappa:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),  
                                                                                          np.mean(out[0]), 
                                                                                          np.mean(out[1]), 
                                                                                          np.mean(out[2]),
                                                                                          np.mean(out[3]),
                                                                                          np.mean(out[4]),))
        f.write('\n') 
        f.write('Time:{},Micro eva,precision:{},oa:{},recall:{},f1:{},kappa:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),  
                                                                                          pre, 
                                                                                          oa, 
                                                                                          re,
                                                                                          f1,
                                                                                          kappa,))
        f.write('\n')      
    print('Testing finished') 
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--net_cfg', type=str, default='SViT_E1_D1_16',help='choose a TransCD model')
    parser.add_argument(
        '--train_cfg', type=str, default='CDNet_2014',help='CDNet_2014 or VL_CMU_CD')
    parser.add_argument(
        '--save_changemap', default=False)
    opt = parser.parse_args()
    
    main(opt)
    
    
    
    
