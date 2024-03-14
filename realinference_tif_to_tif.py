import os
import argparse
import json

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
parser = argparse.ArgumentParser(description='inference')
parser.add_argument('model_name')
args   = parser.parse_args()

image_folder = '_wakelabdata_processed/'
image_name   = "1_Spine_structure_AD_175-11w-D3-xyz6-020C2-T1.tif"
save_folder  = "_result_tif/"
model_name           = args.model_name
configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"]

params["reconstruct"]     = True
params["apply_vq"]        = True
params["use_x_quantized"] = True

JNet = model.JNet(params)
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()

image      = tifpath_to_tensor(os.path.join(image_folder, image_name), False)
print(image.shape)
array_to_tif(os.path.join(save_folder,  model_name+image_name), image.numpy())
crop_size  = (40, 112, 112)
overlap    = ( 0,  0,  0)
# image padding
z_stride   = crop_size[0] - overlap[0]
x_stride   = crop_size[1] - overlap[1]
y_stride   = crop_size[2] - overlap[2]
image_size = image.shape[1:]
scale = params["scale"]
def padsize(image_size, crop_size, overlap_size):
    stride = crop_size - overlap_size
    n   = (image_size - crop_size) // stride + 1
    pad = n * stride + crop_size - image_size
    return pad

zpad  = padsize(image_size[0], crop_size[0], overlap[0])
xpad  = padsize(image_size[1], crop_size[1], overlap[1])
ypad  = padsize(image_size[2], crop_size[2], overlap[2])

image = F.pad(image, (0, ypad, 0, xpad, 0, zpad), "reflect", 0.)
print(image.shape)
result = torch.zeros((1, params["scale"]*image.shape[1], *image.shape[-2:]))
for _z in range(image.shape[1] // (crop_size[0] - overlap[0])):
    for _x in range(image.shape[2] // (crop_size[1] - overlap[1])):
        for _y in range(image.shape[3] // (crop_size[2 ] - overlap[2])):
            crop = image[:, z_stride*_z : z_stride*(_z + 1),
                            x_stride*_x : x_stride*(_x + 1),
                            y_stride*_y : y_stride*(_y + 1),].clone().to(device).unsqueeze(0)
            
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
array_to_tif(os.path.join(save_folder,  model_name+"_re_"+image_name), result)
