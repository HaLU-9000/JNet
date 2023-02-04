from pathlib import Path
import numpy as np
import torch
from utils import save_dataset, save_label

from dataset import Blur
"""
blurring
"""

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Building data on device {device}.")
scale = 10
if scale == 10:
    model = Blur(scale   = scale ,
                 z       = 101   ,
                 x       = 23     ,
                 y       = 23    ,
                 mu_z    = 0.2  ,
                 sig_z   = 0.2   , 
                 bet_xy  = 4.43864   ,
                 bet_z   = 27.7052  ,
                 alpha   = 74.9664,
                 sig_eps = 0.001 ,
                 device  = device,)
    model.eval()
    folderpath    = 'beadslikedataset2'
    outfolderpath = 'beadslikedata5'
    labelname     = '_label'
    outlabelname  = '_label'
    save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, device, 0)