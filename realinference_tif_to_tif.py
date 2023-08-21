import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import RealDensityDataset, sequentialflip

import model_new as model
from utils import tifpath_to_tensor

"input: tif, output: tif"

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Inference on device {device}.")

model_name           = 'JNet_292_fft_finetuning_bead'
params               = {"mu_z"       : 0.2               ,
                        "sig_z"      : 0.2               ,
                        "log_bet_z"  : np.log(30.).item(),
                        "log_bet_xy" : np.log(3.).item() ,
                        "log_alpha"  : np.log(1.).item() ,
                        "sig_eps"    : 0.01              ,
                        "scale"      : 10                ,
                        }

JNet = model.JNet(hidden_channels_list  = [16, 32, 64, 128, 256],
                  nblocks               = 2                     ,
                  activation            = nn.ReLU(inplace=True),
                  dropout               = 0.5                  ,
                  params                = params               ,
                  superres              = True                 ,
                  reconstruct           = True                 ,
                  apply_vq              = True                 ,
                  use_x_quantized       = True                 ,
                  use_fftconv           = True,
                  z = 161, x = 31, y = 31,
                  )
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()

image_path = 'images_psj/'
image      = tifpath_to_tensor(image_path)
crop_size  = (24, 112, 112)
overlap    = ( 1,  10,  10)
# image padding
z_stride   = crop_size[0] - overlap[0]
x_stride   = crop_size[1] - overlap[1]
y_stride   = crop_size[2] - overlap[2]
image_size = image.shape[1:]
zpad = image_size[0] % (crop_size[0] - overlap[0])
xpad = image_size[1] % (crop_size[1] - overlap[1])
ypad = image_size[2] % (crop_size[2] - overlap[2])
result = torch.zeros((0, params["scale"]*image.shape[1], *image.shape[-2:]))
for _z in range(image_size[0] // (crop_size[0] - overlap[0])):
    for _x in range(image_size[1] // (crop_size[1] - overlap[1])):
        for _y in range(image_size[2] // (crop_size[1] - overlap[2])):
            crop = image[0, z_stride*_z : z_stride*_z + crop_size[0],
                            x_stride*_x : x_stride*_x + crop_size[1],
                            y_stride*_y : y_stride*_y + crop_size[2],].clone().to(device)
            crop = JNet(crop)["enhanced_image"] # inference
            result[0, z_stride*_z : z_stride*_z + crop_size[0],
                      x_stride*_x : x_stride*_x + crop_size[1],
                      y_stride*_y : y_stride*_y + crop_size[2],] += crop # sum
            if _z != 0 and overlap[0] != 0: # resolve overlap
                result[0, z_stride*_z : z_stride*_z + overlap[0]  ,
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + crop_size[2],] \
              = result[0, z_stride*_z : z_stride*_z + overlap[0]  ,
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + crop_size[2],].clone() * 1/2
            if _x != 0 and overlap[1] != 0:
                result[0, z_stride*_z : z_stride*_z + crop_size[0],
                          x_stride*_x : x_stride*_x + overlap[1]  ,
                          y_stride*_y : y_stride*_y + crop_size[2],] \
              = result[0, z_stride*_z : z_stride*_z + crop_size[0],
                          x_stride*_x : x_stride*_x + overlap[1]  ,
                          y_stride*_y : y_stride*_y + crop_size[2],].clone() * 1/2
            if _y != 0 and overlap[2] != 0:
                result[0, z_stride*_z : z_stride*_z + crop_size[0],
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + overlap[2]  ,] \
              = result[0, z_stride*_z : z_stride*_z + crop_size[0],
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + overlap[2]  ,].clone() * 1/2
result = result