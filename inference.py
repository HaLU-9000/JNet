import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import RandomCutDataset
import model
import old_model
from dataset import Vibrate

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]
model_name           = 'JNet_268_vibration'
hidden_channels_list = [16, 32, 64, 128, 256]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False
params               = {"mu_z"   : 0.2  , 
                        "sig_z"  : 0.2  , 
                        "log_bet_z"  :  20. , 
                        "log_bet_xy" :   1. , 
                        "log_alpha"  :   1. ,
                        "log_k"      :   0. ,
                        "log_l"      :   0. ,
                        "sig_eps":  0.01,
                        "scale"  :  6
                        }
# must set same param num for training, because of the param estimation layer (will be deleted)
image_size = (1, 1, 240,  96,  96)
original_cropsize = [360, 120, 120]
param_estimation_list = [False, False, False, False, True]

JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  params                = params               ,
                  param_estimation_list = param_estimation_list,
                  superres              = superres             ,
                  reconstruct           = False                ,
                  apply_vq              = True                 ,
                  use_x_quantized       = True             
                  )
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)

val_dataset   = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_30_hill'   ,  ###
                                 imagename   =  f'_x{scale}'            ,     ## scale
                                 labelname   =  '_label'                ,
                                 size        =  (1200, 500, 500)        ,
                                 cropsize    =  ( 240, 112, 112)        ,
                                 I             =  20                    ,
                                 low           =  19                    ,
                                 high          =  20                    ,
                                 scale         =  scale                 ,   ## scale
                                 train         =  False                 ,
                                 mask          =  False                 ,
                                 surround      =  surround              ,
                                 surround_size =  surround_size         ,
                                 seed          =  907                   ,
                                ) 
j = 120
i = 64
j_s = j // scale

val_loader  = DataLoader(val_dataset                   ,
                         batch_size  = 1               ,
                         shuffle     = False           ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )

vibrate = Vibrate()
JNet.eval()
print([i for i in JNet.parameters()][-4:])
figure = True
for n, val_data in enumerate(val_loader):
    if n >= 5:
        break
    image = val_data[0].to(device = device)
    label = val_data[1].to(device = device)
    JNet.set_upsample_rate(params["scale"])
    image = vibrate(image)
    outdict = JNet(image)
    output  = outdict["enhanced_image"]
    reconst = outdict["reconstruction"]
    qloss   = outdict["quantized_loss"]
    est_params = outdict["blur_parameter"]
    _image  = image[0].detach().cpu().numpy()
    _label  = label[0].detach().cpu().numpy()
    _output = output[0].detach().cpu().numpy()
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
        ax1.imshow(_image[0, j_s, :, :],
                cmap='gray', vmin=0.0, aspect=1)
        ax2.imshow(_output[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(_label[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax4.imshow(_image[0, :, i, :],
                cmap='gray', vmin=0.0, aspect=scale)
        ax5.imshow(_output[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax6.imshow(_label[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        plt.savefig(f'result/{model_name}_vq_result_{n}.png',
                    format='png', dpi=250)