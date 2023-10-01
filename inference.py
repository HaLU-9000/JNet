import os
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

model_name            = 'JNet_346_gaussianpsf_finetuning'
pretrained_model_name = 'JNet_344_gaussianpsf_pretrain_woattn'

params     = {"hidden_channels_list"  : [4, 8, 16, 32, 64]                ,
              "attn_list"             : [False, False, False, False, False],     
              "nblocks"               : 2                                 ,     
              "activation"            : nn.ReLU(inplace=True)             ,     
              "dropout"               : 0.5                               ,     
              "superres"              : True                              ,     
              "partial"               : None                              ,
              "reconstruct"           : True                              ,     
              "apply_vq"              : True                              ,     
              "use_fftconv"           : True                              ,     
              "use_x_quantized"       : True                              ,     
              "mu_z"                  : 0.1                               ,
              "sig_z"                 : 0.1                               ,
              "blur_mode"             : "gaussian"                        , # "gaussian" or "gibsonlanni"
              "size_x"                : 51                                ,
              "size_y"                : 51                                ,
              "size_z"                : 161                               ,
              "NA"                    : 0.80                              , # # # # param # # # #
              "wavelength"            : 0.910                             , # microns # # # # param # # # #
              "M"                     : 25                                , # magnification # # # # param # # # #
              "ns"                    : 1.4                               , # specimen refractive index (RI)
              "ng0"                   : 1.5                               , # coverslip RI design value
              "ng"                    : 1.5                               , # coverslip RI experimental value
              "ni0"                   : 1.5                               , # immersion medium RI design value
              "ni"                    : 1.5                               , # immersion medium RI experimental value
              "ti0"                   : 150                               , # microns, working distance (immersion medium thickness) design value
              "tg0"                   : 170                               , # microns, coverslip thickness design value
              "tg"                    : 170                               , # microns, coverslip thickness experimental value
              "res_lateral"           : 0.05                              , # microns # # # # param # # # #
              "res_axial"             : 0.05                              , # microns # # # # param # # # #
              "pZ"                    : 0                                 , # microns, particle distance from coverslip
              "bet_z"                 : 30.                               ,
              "bet_xy"                :  3.                               ,
              "sig_eps"               : 0.01                              ,
              "scale"                 : 6                                ,
              "device"                : device                            ,
              }

JNet = model.JNet(params)
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)

val_dataset   = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_30_fft_blur'   ,  ###
                                 imagename   =  f'_x{params["scale"]}'            ,     ## scale
                                 labelname   =  '_label'                ,
                                 size        =  (1200, 500, 500)        ,
                                 cropsize    =  ( 240, 112, 112)        ,
                                 I             =  20                    ,
                                 low           =  19                    ,
                                 high          =  20                    ,
                                 scale         =  params["scale"]                ,   ## scale
                                 train         =  False                 ,
                                 mask          =  False                 ,
                                 surround      =  False              ,
                                 surround_size =  [32, 4, 4]            ,
                                 seed          =  907                   ,
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
        plt.savefig(f'result/{model_name}_result_{n}.png',
                    format='png', dpi=250)