from pathlib import Path
import numpy as np
import torch
from utils import save_dataset

from dataset import Blur
"""
blurring
"""

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")
for scale in (1, 2, 8):
    model = Blur(scale   = scale ,
                 z       = 141   ,
                 x       = 7     ,
                 y       = 7     ,
                 mu_z    = 0.2   ,
                 sig_z   = 0.2   , 
                 bet_xy  = 6.    ,
                 bet_z   = 35.   ,
                 sig_eps = 0.02  ,)
    model.eval
    folderpath    = 'randomdataset'
    outfolderpath = 'randomdata'
    labelname     = '_label'
    outlabelname  = '_label'
    save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale)