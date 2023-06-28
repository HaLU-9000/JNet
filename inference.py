import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import LabelandBlurParamsDataset, gen_imaging_parameters, Augmentation
import model

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]


model_name           = 'JNet_219_x6'
hidden_channels_list = [16, 32, 64, 128, 256]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False

params               = {"mu_z"   : 0.2    ,
                        "sig_z"  : 0.2    ,
                        "bet_z"  : 20.    ,
                        "bet_xy" : 1.0    ,
                        "alpha"  : 1.0    ,
                        "sig_eps": 0.01   ,
                        "scale"  : 6
                        }

params_ranges = {"mu_z"   : [0,   1, 0.2  ,  0.0001 ],
                 "sig_z"  : [0,   1, 0.2  ,  0.0001 ],
                 "bet_z"  : [0 , 22,  20  ,  0.0001 ],
                 "bet_xy" : [0,   2,   1. ,  0.0001 ],
                 "alpha"  : [0,   2,   1. ,  0.0001 ],
                 "sig_eps": [0, 0.012, 0.01, 0.0001 ],
                 "scale"  : [6]
                 }

param_scales = {"mu_z"   :  1,
                "sig_z"  :  1,
                "bet_z"  : 22,
                "bet_xy" :  2,
                "alpha"  :  2,}

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
                  apply_vq              = False                ,
                  )
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)

val_dataset   = LabelandBlurParamsDataset(folderpath           = "beadslikedataset2"                  ,
                                          size                 = (1200, 500, 500)                         ,
                                          cropsize             = original_cropsize                        ,
                                          I                    = 20                                       ,
                                          low                  = 0                                       ,
                                          high                 = 1                                       ,
                                          imaging_function     = JNet.image                               ,
                                          imaging_params_range = params_ranges                            ,
                                          validation_params    = gen_imaging_parameters(params_ranges)    ,
                                          device               = device                                   ,
                                          is_train             = False                                    ,
                                          mask                 = False                                    ,
                                          surround             = surround                                 ,
                                          surround_size        = surround_size                            ,
                                          seed                 = 907                                      ,
                                          )
j = 120
i = 60
j_s = j // scale

augment_param     = {"mask"          : False            , 
                     "mask_size"     : [1, 10, 10]      , 
                     "mask_num"      : 30               , 
                     "surround"      : surround         , 
                     "surround_size" : surround_size    ,
                     "original_size" : original_cropsize,
                     "cropsize"      : image_size[2:]   ,}

val_augment = Augmentation(augment_param)

val_loader  = DataLoader(val_dataset                   ,
                         batch_size  = 4               ,
                         shuffle     = False           ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )

JNet.eval()
count = 0
figure = False
for val_data in val_loader:
    
    label  = val_data[0].to(device = device)
    params = val_data[1]
    image  = JNet.image.sample_from_params(label, params).float()
    image, label = val_augment.crop(image, label)
    JNet.set_upsample_rate(params["scale"][0])
    outdict= JNet(image)
    output = outdict["enhanced_image"]
    reconst= outdict["reconstruction"]
    qloss  = outdict["quantized_loss"]
    est_params = outdict["blur_parameter"]
    lossfunc = nn.BCELoss()
    print(lossfunc(output.detach().cpu(), label.detach().cpu()))
    num = image.shape[0]
    for n in range(num):
        _image   = image[n].detach().cpu().numpy()
        _label   = label[n].detach().cpu().numpy()
        _output  = output[n].detach().cpu().numpy()
        _reconst = reconst[n].detach().cpu().numpy()
        fig = plt.figure(figsize=(25, 15))
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)
        #ax7 = fig.add_subplot(247)
        #ax8 = fig.add_subplot(248)
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax4.set_axis_off()
        ax5.set_axis_off()
        ax6.set_axis_off()
        #ax7.set_axis_off()
        #ax8.set_axis_off()
        #ax1.set_title('plain\nreconstruct image')
        #ax2.set_title('plain\noriginal image')
        #ax3.set_title('plain\nsegmentation result')
        #ax4.set_title('plain\nlabel')
        #ax5.set_title('depth\nreconstruct')
        #ax6.set_title('depth\noriginal image')
        #ax7.set_title('depth\nsegmentation result')
        #ax8.set_title('depth\nlabel')
        plt.subplots_adjust(hspace=-0.0)
        if figure:
            if partial is not None:
                ax1.imshow(_reconst[0, partial[0]+j_s, :, :],
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                ax2.imshow(_image[0, partial[0]+j_s, :, :].to(device='cpu'),
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                ax3.imshow(_output[0, 0, partial[0]+j_s, :, :],
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                ax4.imshow(_label[0, partial[0]+j, :, :].to(device='cpu'),
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                ax5.imshow(_reconst[0, partial[0]:partial[1], i, :],
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
                ax6.imshow(_image[0, partial[0]:partial[1], i, :].to(device='cpu'),
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
                #ax7.imshow(output[0, 0, partial[0]:partial[1], i, :],
                #        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                #ax8.imshow(label[0, partial[0]:partial[1], i, :].to(device='cpu'),
                #        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            else:
                #ax1.imshow(reconst[0, j_s, :, :],
                #        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                ax1.imshow(_image[0, j_s, :, :],
                        cmap='gray', vmin=0.0, aspect=1)
                ax2.imshow(_output[0, j, :, :],
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                ax3.imshow(_label[0, j, :, :],
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                #ax5.imshow(reconst[0, :, i, :],
                #        cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
                ax4.imshow(_image[0, :, i, :],
                        cmap='gray', vmin=0.0, aspect=scale)
                ax5.imshow(_output[0, :, i, :],
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
                ax6.imshow(_label[0, :, i, :],
                        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            plt.savefig(f'result/{model_name}_train0result_{n}.png', format='png', dpi=250)
        count += 1
