#!/usr/bin/python
# -*- coding: UTF-8 -*-
import ml_collections
from llvmlite.ir.instructions import Resume
from astropy.visualization.tests.test_lupton_rgb import display


def CDNet_2014_training_config():
    """Returns training configuration."""
    
    config = ml_collections.ConfigDict()
    config.dataset_name = 'CDNet_2014'
    config.category = ['badWeather', 'baseline', 'cameraJitter', 'dynamicBackground', 'intermittentObjectMotion', 
'lowFramerate', 'nightVideos', 'PTZ', 'shadow', 'thermal', 'turbulence']
    config.resume = False # if resume training
    config.display = True # if display images and loss curve in Visdom
    config.print_rate = 20
    config.im_size = 512
    config.batch_size = 16
    config.epoch = 20
    
    
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.lr = 0.0002
    config.optimizer.momentum = 0.9
    config.optimizer.weight_decay = 0.0005
    config.optimizer.lr_step_size = 50
    
    config.path = ml_collections.ConfigDict()
    config.path.dataset = 'G:/dataset/ChangeDetection/cd2014/dataset' # change to your dataset path
    config.path.train_txt = './CDNet_2014/data/train.txt'
    config.path.test_txt = './CDNet_2014/data/test.txt'
    config.path.val_txt = './CDNet_2014/data/val.txt'
    config.path.txt = './CDNet_2014/data'
    config.path.outputs = './CDNet_2014/outputs'
    config.path.weights_save_dir = './CDNet_2014/outputs/weights'
    config.path.best_weights_save_dir = './CDNet_2014/outputs/best_weights'
    config.path.changemap = './CDNet_2014/outputs/changemap'
    
    return config

def VL_CMU_CD_training_config():
    """Returns training configuration."""
    
    config = ml_collections.ConfigDict()
    config.dataset_name = 'VL_CMU_CD'
    config.category = ['test']
    config.resume = False # if resume training
    config.display = True # if display images and loss curve in Visdom
    config.print_rate = 20
    config.im_size = 512
    config.batch_size = 4
    config.epoch = 50
    
    
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.lr = 0.0005
    config.optimizer.momentum = 0.9
    config.optimizer.weight_decay = 0.0005
    config.optimizer.lr_step_size = 50
    
    config.path = ml_collections.ConfigDict()
    config.path.dataset = 'G:/dataset/ChangeDetection/VL-CMU-CD/VL-CMU-CD_dataset/raw' # change to your dataset path
    config.path.train_txt = './VL_CMU_CD/data/train.txt'
    config.path.test_txt = './VL_CMU_CD/data/test.txt'
    config.path.val_txt = './VL_CMU_CD/data/val.txt'
    config.path.txt = './VL_CMU_CD/data'
    config.path.outputs = './VL_CMU_CD/outputs'
    config.path.weights_save_dir = './VL_CMU_CD/outputs/weights'
    config.path.best_weights_save_dir = './VL_CMU_CD/outputs/best_weights'
    config.path.changemap = './VL_CMU_CD/outputs/changemap'
    
    return config    
    
def ViT_E1_D1_16_config():
    
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.feature_fusion_channel = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 1
    config.transformer.decoder_num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    
    config.if_use_backbone = False
    config.backbone = 'ResNet18'   # ResNet18, 34, 50, 101, 152
    config.patches = ml_collections.ConfigDict() 
    config.patches.grid = (16, 16) 
        # Using resnet as embbedding, Tokens size = grid
    config.resnet = ml_collections.ConfigDict()
    config.resnet.downsampling_rate = 16      #the downsampling rate of the used resnet or other backbone

    return config

def ViT_E1_D1_32_config():
    
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.feature_fusion_channel = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 1
    config.transformer.decoder_num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    
    config.if_use_backbone = False
    config.backbone = 'ResNet18'   # ResNet18, 34, 50, 101, 152
    config.patches = ml_collections.ConfigDict() 
    config.patches.grid = (32, 32) 
        # Using resnet as embbedding, Tokens size = grid
    config.resnet = ml_collections.ConfigDict()
    config.resnet.downsampling_rate = 16      #the downsampling rate of the used resnet or other backbone

    return config

def ViT_E4_D4_16_config():
    
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.feature_fusion_channel = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 4
    config.transformer.decoder_num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    
    config.if_use_backbone = False
    config.backbone = 'ResNet18'   # ResNet18, 34, 50, 101, 152
    config.patches = ml_collections.ConfigDict() 
    config.patches.grid = (16, 16) 
        # Using resnet as embbedding, Tokens size = grid
    config.resnet = ml_collections.ConfigDict()
    config.resnet.downsampling_rate = 16      #the downsampling rate of the used resnet or other backbone

    return config

def ViT_E4_D4_32_config():
    
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.feature_fusion_channel = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 4
    config.transformer.decoder_num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    
    config.if_use_backbone = False
    config.backbone = 'ResNet18'   # ResNet18, 34, 50, 101, 152
    config.patches = ml_collections.ConfigDict() 
    config.patches.grid = (32, 32) 
        # Using resnet as embbedding, Tokens size = grid
    config.resnet = ml_collections.ConfigDict()
    config.resnet.downsampling_rate = 16      #the downsampling rate of the used resnet or other backbone

    return config

def Res_ViT_E1_D1_16_config():
    
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.feature_fusion_channel = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 1
    config.transformer.decoder_num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    
    config.if_use_backbone = True
    config.backbone = 'ResNet18'   # ResNet18, 34, 50, 101, 152
    config.patches = ml_collections.ConfigDict() 
    config.patches.grid = (16, 16) 
        # Using resnet as embbedding, Tokens size = grid
    config.resnet = ml_collections.ConfigDict()
    config.resnet.downsampling_rate = 16      #the downsampling rate of the used resnet or other backbone

    return config

def Res_ViT_E1_D1_32_config():
    
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.feature_fusion_channel = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 1
    config.transformer.decoder_num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    
    config.if_use_backbone = True
    config.backbone = 'ResNet18'   # ResNet18, 34, 50, 101, 152
    config.patches = ml_collections.ConfigDict() 
    config.patches.grid = (32, 32) 
        # Using resnet as embbedding, Tokens size = grid
    config.resnet = ml_collections.ConfigDict()
    config.resnet.downsampling_rate = 16      #the downsampling rate of the used resnet or other backbone

    return config

def Res_ViT_E4_D4_16_config():
    
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.feature_fusion_channel = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 4
    config.transformer.decoder_num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None
    
    config.if_use_backbone = True
    config.backbone = 'ResNet18'   # ResNet18, 34, 50, 101, 152
    config.patches = ml_collections.ConfigDict() 
    config.patches.grid = (16, 16) 
        # Using resnet as embbedding, Tokens size = grid
    config.resnet = ml_collections.ConfigDict()
    config.resnet.downsampling_rate = 16      #the downsampling rate of the used resnet or other backbone

    return config

CONFIGS = {
    'CDNet_2014': CDNet_2014_training_config(),
    'VL_CMU_CD': VL_CMU_CD_training_config(),
    'SViT_E1_D1_16': ViT_E1_D1_16_config(),
    'SViT_E1_D1_32': ViT_E1_D1_32_config(),
    'SViT_E4_D4_16': ViT_E4_D4_16_config(),
    'SViT_E4_D4_32': ViT_E4_D4_32_config(),
    'Res_SViT_E1_D1_16': Res_ViT_E1_D1_16_config(),
    'Res_SViT_E1_D1_32': Res_ViT_E1_D1_32_config(),
    'Res_SViT_E4_D4_16': Res_ViT_E4_D4_16_config(),
}
 



