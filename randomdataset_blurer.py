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
print(f"Training on device {device}.")
scale = 10
if scale == 10:
    model = Blur(scale   = scale ,
                 z       = 71    ,
                 x       = 5     ,
                 y       = 5     ,
                 mu_z    = -2.114,
                 sig_z   = 0.675 , 
                 bet_xy  = 3     ,
                 bet_z   = 17.5  ,
                 sig_eps = 0.01  ,)
    model.eval
    folderpath    = 'beadslikedataset2'
    outfolderpath = 'beadslikedata3'
    labelname     = '_label'
    outlabelname  = '_label'
    save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, 0)