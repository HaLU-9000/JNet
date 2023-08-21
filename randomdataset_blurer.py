from pathlib import Path
import numpy as np
import torch
from utils import save_dataset, save_label

from model_new import ImagingProcess
"""
blurring
"""

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Building data on device {device}.")
scale = 6
params = {"mu_z"       :  0.2             , 
          "sig_z"      :  0.2             , 
          "log_bet_z"  :  np.log(30.).item() , 
          "log_bet_xy" :  np.log(3.).item() , 
          "log_k"      : np.log(1.).item(),
          "log_l"      : np.log(1.).item(),
          "sig_eps"    :  0.02            ,
          "scale"      :  6               ,
          }
if scale == 6:
    model = ImagingProcess(device      = device     ,
                           params      = params     , 
                           z           = 161        ,
                           x           = 31         ,
                           y           = 31         ,
                           dist        = "gaussian" ,
                           apply_hill  = False      ,
                           use_fftconv = True       ,
                           )
    model.eval()
    folderpath    = '_var_num_beadsdataset2'
    outfolderpath = '_var_num_beadsdata2_30_fft_blur'
    labelname     = '_label'
    outlabelname  = '_label'
    save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, device, 0)