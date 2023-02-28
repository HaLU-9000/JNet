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
if scale == 6:
    model = ImagingProcess(mu_z     = 0.2    ,
                           sig_z    = 0.2    ,
                           scale    = [scale, 1, 1],
                           device   = device ,
                           z        = 161    ,
                           x        = 3      ,
                           y        = 3      ,
                           bet_z    = 23.5329,
                           bet_xy   = 1.00000,
                           alpha    = 0.9544 ,
                           sig_eps  = 0.01   ,
                           )
    model.eval()
    folderpath    = 'beadslikedataset2'
    outfolderpath = 'spinelikedata0'
    labelname     = '_label'
    outlabelname  = '_label'
    save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, device, 0)