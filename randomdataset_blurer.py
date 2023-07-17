from pathlib import Path
import numpy as np
import torch
from utils import save_dataset, save_label

from model import ImagingProcess
"""
blurring
"""

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Building data on device {device}.")
scale = 6
params = {"mu_z"   : 0.2  , 
          "sig_z"  : 0.2  , 
          "bet_z"  :  20. , 
          "bet_xy" :   1. , 
          "alpha"  :   1. , 
          "sig_eps":  0.02,
          "scale"  :  6
          }
if scale == 6:
    model = ImagingProcess(device   = device ,
                           params   = params , 
                           z        = 161    ,
                           x        = 3      ,
                           y        = 3      ,
                           )
    model.eval()
    folderpath    = '_var_num_beadsdataset2'
    outfolderpath = '_var_num_beadsdata2_20'
    labelname     = '_label'
    outlabelname  = '_label'
    save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, device, 0)