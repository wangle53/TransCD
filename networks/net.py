#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import torch
import torch.nn as nn
sys.path.append('./networks')
import configs as cfg
from vision_transformer import CD_Transformer as CDTR



class Feature_Fusion_Head(nn.Module):
    def __init__(self, config, img_size, out_channel=1):
        super(Feature_Fusion_Head, self).__init__()
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            self.H =  grid_size[0]
            self.W =  grid_size[1]
        else:
            self.H = img_size // config.patches["size"][0]
            self.W = img_size // config.patches["size"][1]
        down_sample_rate = img_size//self.H
        self.hidden_size = config.hidden_size
        ffc = config.feature_fusion_channel
        
        self.ff_layer = nn.Sequential(
            nn.Conv2d(self.hidden_size, ffc, 3, 1, 1),
            nn.BatchNorm2d(ffc),
            nn.ReLU(inplace=True),
            )
        self.up_layer = nn.Sequential()
        while down_sample_rate // 2 != 1:
            self.up_layer.add_module('up-layers-{0}-{1}-conv'.format(ffc, ffc//2),
                nn.ConvTranspose2d(ffc, ffc//2, 4, 2, 1))
            self.up_layer.add_module('up-layers-{0}-{1}-bn'.format(ffc, ffc//2),
                nn.BatchNorm2d(ffc//2))
            self.up_layer.add_module('up-layers-{0}-{1}_relu'.format(ffc, ffc//2),
                nn.ReLU(inplace=True))    
            down_sample_rate //= 2
            ffc //= 2

        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(ffc, out_channel, 4, 2, 1),
                nn.BatchNorm2d(out_channel),
                nn.Sigmoid(),
#                 nn.ReLU(inplace=True),
            )
        
    
    def forward(self, x1_features, x2_features):
        B, HW, C = x1_features.shape

        x1 = x1_features.permute(0,2,1).view(B, C, self.H, self.W)
        x2 = x2_features.permute(0,2,1).view(B, C, self.H, self.W)
#         x = torch.cat((x1, x2),1)
        x = torch.abs(x1-x2)
        x = self.ff_layer(x)
        x = self.up_layer(x)
        x = self.last_layer(x)
        return x

class TransCDNet(nn.Module):
    def __init__(self, config, img_size, vis):
        super(TransCDNet, self).__init__()
        self.cdtr = CDTR(config, img_size, vis)  
        self.ffn = Feature_Fusion_Head(config, img_size, 1)
    def forward(self, x1, x2):
        x1 = self.cdtr(x1)
        x2 = self.cdtr(x2)
        out = self.ffn(x1[0], x2[0])
        return out
    
    
def main():
    input = torch.randn(2, 3, 512,512)
    net = TransCDNet(cfg.get_r18_e1d8h16_config(), 512, False)
#     print(net)
    y = net(input,input)
    print(y.shape)
    
if __name__ == '__main__':
    main()
