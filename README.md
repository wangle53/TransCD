# [TransCD: Scene Change Detection via Transformer-based Architecture](https://doi.org/10.1364%2Foe.440720)
![image](https://user-images.githubusercontent.com/79884379/140607552-c42c612d-fe9b-40c6-830b-404c3d25f9c0.png)
## Requirements
```
Python 3.7.0  
Pytorch 1.6.0  
Visdom 0.1.8.9  
Torchvision 0.7.0
```
## Datasets
- CD2014 dataset
  - paper: [changedetection.net: A new change detection benchmark dataset](https://www.merl.com/publications/docs/TR2012-044.pdf)
  - paper: [CDnet 2014: An Expanded Change Detection Benchmark Dataset](https://www.cv-foundation.org//openaccess/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf)
  - dataset: http://changedetection.net/
  - if the above link didn't work, please try this baidudisk link: [[Baiduyun]](https://pan.baidu.com/s/1vccmmWAcp6kVLZCam_Av8g) the password is: ob6j
- VL-CMU-CD
  - paper: [Street-view change detection with deconvolutional networks](http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf)
  - dataset:  https://ghsi.github.io/proj/RSS2016.html
  - if the above link didn't work, please try this baidudisk link: [[Baiduyun]](https://pan.baidu.com/s/1-PfS4P8Fij_dkjJXpLkHUQ) the password is: v82c
  

## Pretrained Model
Pretrained models for CDNet-2014 and VL-CMU-CD are available. You can download them from the following link.
- CDNet-2014: [[Baiduyun]](https://pan.baidu.com/s/16h6uEMgEkCAJdDZa7lfbag) the password is 78cp. [[GoogleDrive](https://drive.google.com/drive/folders/1bKnTMXRx0Og8lBV7jNC0PdoeKfJ5GeUn?usp=sharing)].
  - We uploaded six models trained on CDNet-2014 dataset, they are SViT_E1_D1_16, SViT_E1_D1_32, SViT_E4_D4_16, SViT_E4_D4_32, Res_SViT_E1_D1_16 and Res_SViT_E4_D4_16.
- VL-CMU-CD: [[Baiduyun](https://pan.baidu.com/s/1vDKWCW3dO-JX_ET4OgsNww)] the password is ydzl. [[GoogleDrive](https://drive.google.com/drive/folders/1-6BvZLtSbu96cjAW7KpDNNrwPhJz99V3?usp=sharing)].
  - We uploaded four models trained on VL-CMU-CD dataset, they are SViT_E1_D1_16, SViT_E1_D1_32, Res_SViT_E1_D1_16 and Res_SViT_E1_D1_32.
## Test
Before test, please download datasets and pretrained models. Copy pretrained models to folder './dataset_name/outputs/best_weights', and run the following command: 
```
cd TransCD_ROOT
python test.py --net_cfg <net name> --train_cfg <training configuration>
```  
Use `--save_changemap True` to save predicted changemaps.
For example:
```
python test.py --net_cfg SViT_E1_D1_32 --train_cfg CDNet_2014 --save_changemap True
```

## Training
Before training, please download datasets and revise dataset path in configs.py to your path.
CD TransCD_ROOT
```
python -m visdom.server
python train.py --net_cfg <net name> --train_cfg <training configuration>
```
For example:
```
python -m visdom.server
python train.py --net_cfg Res_SViT_E1_D1_16 --train_cfg VL_CMU_CD
```
To display training processing, open 'http://localhost:8097' in your browser.
## Citing TransCD
If you use this repository or would like to refer the paper, please use the following BibTex entry.
```
@article{Wang:21,
author = {Zhixue Wang and Yu Zhang and Lin Luo and Nan Wang},
journal = {Opt. Express},
keywords = {Feature extraction; Neural networks; Object detection; Segmentation; Spatial resolution; Vision modeling},
number = {25},
pages = {41409--41427},
publisher = {OSA},
title = {TransCD: scene change detection via transformer-based architecture},
volume = {29},
month = {Dec},
year = {2021},
url = {http://www.osapublishing.org/oe/abstract.cfm?URI=oe-29-25-41409},
doi = {10.1364/OE.440720},
}
```
## Reference
```
-Akcay, Samet, Amir Atapour-Abarghouei, and Toby P. Breckon. "Ganomaly: Semi-supervised anomaly detection via adversarial training." Asian conference on computer vision. Springer, Cham, 2018.
-Chen, Jieneng, et al. "Transunet: Transformers make strong encoders for medical image segmentation." arXiv preprint arXiv:2102.04306 (2021).
```
## More
[My personal google web](https://scholar.google.com/citations?user=qdkY0jcAAAAJ&hl=zh-TW)
