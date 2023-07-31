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
params = {"mu_z"       :  0.2             , 
          "sig_z"      :  0.2             , 
          "log_bet_z"  :  np.log(27.7052).item() , 
          "log_bet_xy" :  np.log(4.43864).item() , 
          "log_alpha"  :  np.log(74.9664).item() , 
          "sig_eps"    :  0.02            ,
          "scale"      :  6               ,
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
    outfolderpath = '_var_num_beadsdata2_27'
    labelname     = '_label'
    outlabelname  = '_label'
    save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, device, 0)