import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import RealDensityDataset, sequentialflip

import model_new as model
from utils import tifpath_to_tensor, array_to_tif

"input: tif, output: tif"

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Inference on device {device}.")

image_folder = '_wakelabdata_processed/'
image_name   = "MD495_1G2_D14_FINC1-T1.tif"
save_folder  = "_result_tif/"
model_name           = 'JNet_314_ewc_finetuning'
params               = {"mu_z"       : 0.2               ,
                        "sig_z"      : 0.2               ,
                        "log_bet_z"  : np.log(30.).item(),
                        "log_bet_xy" : np.log(3.).item() ,
                        "log_alpha"  : np.log(1.).item() ,
                        "sig_eps"    : 0.01              ,
                        "scale"      : 12                 ,
                        }

JNet = model.JNet(hidden_channels_list  = [16, 32, 64, 128, 256],
                  nblocks               = 2                     ,
                  activation            = nn.ReLU(inplace=True),
                  dropout               = 0.5                  ,
                  params                = params               ,
                  superres              = True                 ,
                  reconstruct           = True                 ,
                  apply_vq              = True                 ,
                  use_x_quantized       = False                ,
                  use_fftconv           = True,
                  z = 161, x = 31, y = 31,
                  )
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()

image      = tifpath_to_tensor(os.path.join(image_folder, image_name), False)
print(image.shape)
image = image[:, 1:17, 512:1024, 512:1024].clone()
array_to_tif(os.path.join(save_folder,  "original"+model_name+image_name), image.numpy())
crop_size  = (16, 112, 112)
overlap    = ( 1,  10,  10)
# image padding
z_stride   = crop_size[0] - overlap[0]
x_stride   = crop_size[1] - overlap[1]
y_stride   = crop_size[2] - overlap[2]
image_size = image.shape[1:]
scale = params["scale"]
def padsize(image_size, crop_size, overlap_size):
    stride = crop_size - overlap_size
    n   = (image_size - crop_size) // stride + 2
    pad = n * stride + crop_size - image_size
    return pad

zpad  = padsize(image_size[0], crop_size[0], overlap[0])
xpad  = padsize(image_size[1], crop_size[1], overlap[1])
ypad  = padsize(image_size[2], crop_size[2], overlap[2])
image = F.pad(image, (0, ypad, 0, xpad, 0, zpad), "constant", 0.)
print(image.shape)
result = torch.zeros((1, params["scale"]*image.shape[1], *image.shape[-2:]))
for _z in range(image.shape[1] // (crop_size[0] - overlap[0]) - 1):
    for _x in range(image.shape[2] // (crop_size[1] - overlap[1] - 1)):
        for _y in range(image.shape[3] // (crop_size[1] - overlap[2] - 1)):
            crop = image[:, z_stride*_z : z_stride*_z + crop_size[0],
                            x_stride*_x : x_stride*_x + crop_size[1],
                            y_stride*_y : y_stride*_y + crop_size[2],].clone().to(device).unsqueeze(0)
            crop = JNet(crop)["enhanced_image"].squeeze(0).detach().cpu()
            result[:, (z_stride*_z)*scale : (z_stride*_z + crop_size[0]) * scale,
                      x_stride*_x : x_stride*_x + crop_size[1],
                      y_stride*_y : y_stride*_y + crop_size[2],] += crop # sum
            if _z != 0 and overlap[0] != 0: # resolve overlap
                result[:, (z_stride*_z)*scale : (z_stride*_z + overlap[0]) * scale,
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + crop_size[2],] \
              = result[:, (z_stride*_z)*scale : (z_stride*_z + overlap[0]) * scale,
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + crop_size[2],].clone() * 1/2
            if _x != 0 and overlap[1] != 0:
                result[:, (z_stride*_z)*scale : (z_stride*_z + crop_size[0]) * scale,
                          x_stride*_x : x_stride*_x + overlap[1]  ,
                          y_stride*_y : y_stride*_y + crop_size[2],] \
              = result[:, (z_stride*_z)*scale : (z_stride*_z + crop_size[0]) * scale,
                          x_stride*_x : x_stride*_x + overlap[1]  ,
                          y_stride*_y : y_stride*_y + crop_size[2],].clone() * 1/2
            if _y != 0 and overlap[2] != 0:
                result[:, (z_stride*_z)*scale : (z_stride*_z + crop_size[0]) * scale,
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + overlap[2]  ,] \
              = result[:, (z_stride*_z)*scale : (z_stride*_z + crop_size[0]) * scale,
                          x_stride*_x : x_stride*_x + crop_size[1],
                          y_stride*_y : y_stride*_y + overlap[2]  ,].clone() * 1/2
result = result[:, :-zpad*scale, :-xpad, :-ypad].detach().cpu().numpy()
print(result.shape)
array_to_tif(os.path.join(save_folder,  model_name+image_name), result)

