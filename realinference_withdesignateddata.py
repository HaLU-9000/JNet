import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import RealDensityDataset, sequentialflip

import model_new as model

vis_mseloss = False

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Inference on device {device}.")

model_name            = 'JNet_343_physicspsf_pretrain_woattn'
#pretrained_model_name = 'JNet_344_gaussianpsf_pretrain_woattn'

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
              "blur_mode"             : "gibsonlanni"                     , # "gaussian" or "gibsonlanni"
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
              "scale"                 : 10                                ,
              "device"                : device                            ,
              }

JNet = model.JNet(params)
JNet = JNet.to(device = device)
j   = 12
j_s = j // params["scale"]
i = 64

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()
dirpath = "_beads_roi_extracted_stackreg"
images = [os.path.join(dirpath, f) for f in sorted(os.listdir(dirpath))]
print(images)
#losses = []
outputs = []
loss_fn = nn.MSELoss()
for image_name in images[:-1]:
    image_ = torch.load(image_name, map_location="cuda").to(torch.float32)
    image = image_#(torch.clip(image_, min=0.1, max=1.) - 0.1) / (1.0 - 0.1)
    outdict = JNet(image.to("cuda").unsqueeze(0))
    output  = outdict["enhanced_image"]
    output  = output.detach().cpu()
    reconst = outdict["reconstruction"]
    #loss    = loss_fn(reconst, image).item()
    qloss   = outdict["quantized_loss"]
    print("output ", torch.sum(output).item() * (0.05 * 0.05 * 0.05))
    outputs.append(torch.sum(output).item()* (0.05 * 0.05 * 0.05))
    reconst = reconst.squeeze(0).detach().cpu().numpy()
    #losses.append(loss)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.set_title('original image')
    ax2.set_title(f'reconstruct image\n{model_name}')
    plt.subplots_adjust(hspace=-0.0)
    ax1.imshow(image_[0, :, i, :].to(device='cpu'),
            cmap='gray', vmin=0.0, vmax=1.0, aspect=params["scale"])
    ax2.imshow(output[0, 0, :, i, :],
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    plt.savefig(f'result/{model_name}_new_{image_name[30:-3]}.png', format='png', dpi=250)
print(model_name)
print(np.mean(np.array(outputs)))

#print(losses)

# for image_name in images[:-1]:
#     image_ = torch.load(image_name, map_location="cuda")
#     c_, z_, x_, y_ = image_.shape
#     ensenbled_output  = torch.zeros((c_, z_*scale, x_, y_), device='cpu')
#     ensenbled_reconst = torch.zeros((c_, z_, x_, y_), device='cpu')
#     ensenbled_loss    = 0
#     ensenbled_qloss   = 0

#     for _ in range(8):
#         image   = sequentialflip(image_, i)#(torch.clip(image_, min=0.1, max=1.) - 0.1) / (1.0 - 0.1)
#         outdict = JNet(image.to("cuda").unsqueeze(0))
#         output  = outdict["enhanced_image"]
#         output  = output.detach().cpu().squeeze(0)
#         output  = sequentialflip(output.clone(), i)
#         image   = image.cpu()
#         ensenbled_output  += output  / 8
#     print("output ", torch.sum(ensenbled_output) * (0.05 * 0.05 * 0.05))
#     fig = plt.figure(figsize=(10, 10))
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     ax1.set_axis_off()
#     ax2.set_axis_off()
#     ax1.set_title('original image')
#     ax2.set_title(f'reconstruct image\n{model_name}')
#     plt.subplots_adjust(hspace=-0.0)
#     dirpath = "_beads_roi_extracted_stackreg/"
#     #image_ = torch.load(image_name, map_location="cuda")
#     image_ = torch.load(image_name, map_location="cuda")
#     ax1.imshow(image_.cpu().numpy()[0,:,i, :], cmap="gray", vmin=0, vmax=1, aspect=10)
#     ax2.imshow(ensenbled_output.cpu().numpy()[0, :, i, :],
#              cmap="gray", vmin=0.0, vmax=1.0, aspect=1)
#     plt.savefig(f'result/{model_name}_new_{image_name[30:-3]}.png', format='png', dpi=250)