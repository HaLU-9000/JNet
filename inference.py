import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import RandomCutDataset
import model_new as model
import old_model
from dataset import Vibrate

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

parser = argparse.ArgumentParser(description='inference for simulation data')
parser.add_argument('model_name')
parser.add_argument("--check_pretrained", default=False)
args   = parser.parse_args()

configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"]

val_dataset_params   = configs["pretrain_val_dataset"]

JNet = model.JNet(params)
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{args.model_name}.pt'), strict=False)

val_dataset   = RandomCutDataset(folderpath    = val_dataset_params["folderpath"]   ,
                                 imagename     = val_dataset_params["imagename"]    , 
                                 labelname     = val_dataset_params["labelname"]    ,
                                 size          = val_dataset_params["size"]         ,
                                 cropsize      = val_dataset_params["cropsize"]     , 
                                 I             = val_dataset_params["I"]            ,
                                 low           = val_dataset_params["low"]          ,
                                 high          = val_dataset_params["high"]         ,
                                 scale         = val_dataset_params["scale"]        ,  ## scale
                                 mask          = val_dataset_params["mask"]         ,
                                 mask_size     = val_dataset_params["mask_size"]    ,
                                 mask_num      = val_dataset_params["mask_num"]     ,  #( 1% of image)
                                 surround      = val_dataset_params["surround"]     ,
                                 surround_size = val_dataset_params["surround_size"],
                                 seed          = val_dataset_params["seed"]         ,
                                ) 
j = 120
i = 64
j_s = j // params["scale"]

val_loader  = DataLoader(val_dataset                   ,
                         batch_size  = 1               ,
                         shuffle     = False           ,
                         pin_memory  = False           ,
                         num_workers = os.cpu_count()  ,
                         )

vibrate = Vibrate()
figure = True
for n, val_data in enumerate(val_loader):
    if n >= 5:
        break
    image = val_data[0].to(device = device)
    print(image.shape)
    label = val_data[1].to(device = device)
    image = vibrate(image).detach().clone()
    outdict = JNet(image)
    output  = outdict["enhanced_image"]
    print(output.shape)
    reconst = outdict["reconstruction"]
    qloss   = outdict["quantized_loss"]
    image  = image[0].detach().cpu().numpy()
    label  = label[0].detach().cpu().numpy()
    output = output[0].detach().cpu().numpy()
    fig = plt.figure(figsize=(25, 15))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    ax5.set_axis_off()
    ax6.set_axis_off()
    plt.subplots_adjust(hspace=-0.0)
    if figure:
        ax1.imshow(image[0, j_s, :, :],
                cmap='gray', vmin=0.0, aspect=1)
        ax2.imshow(output[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(label[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax4.imshow(image[0, :, i, :],
                cmap='gray', vmin=0.0, aspect= params["scale"])
        ax5.imshow(output[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax6.imshow(label[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        plt.savefig(f'result/{args.model_name}_result_{n}.png',
                    format='png', dpi=250)