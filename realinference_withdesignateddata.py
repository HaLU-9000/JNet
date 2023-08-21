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

scale    = 10
surround = False
surround_size = [32, 4, 4]
image_path    = 'images_psj/'
image_name    = 'croped_cliped_beads'
image__name   = 'croped_cliped_beads'
image         = torch.load('./' + image_path + image_name  + '.pt').to(device)
image_        = torch.load('./' + image_path + image__name + '.pt').to(device)

model_name           = 'JNet_292_fft_finetuning_bead'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_factor         = (scale, 1, 1)
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres             = True if scale > 1 else False
reconstruct          = True
params               = {"mu_z"       : 0.2               ,
                        "sig_z"      : 0.2               ,
                        "log_bet_z"  : np.log(30.).item(),
                        "log_bet_xy" : np.log(3.).item() ,
                        "log_alpha"  : np.log(1.).item() ,
                        "sig_eps": 0.01                  ,
                        "scale"  : 10                    ,
                        }             
reconstruct = True
param_estimation_list = [False, False, False, False, True]
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  params                = params               ,
                  superres              = superres             ,
                  reconstruct           = reconstruct          ,
                  apply_vq              = True                 ,
                  use_x_quantized       = True                 ,
                  use_fftconv           = True,
                  z = 161, x = 31, y = 31,
                  )

JNet = JNet.to(device = device)
j   = 12
j_s = j // scale
i = 64

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
ez0, bet_z, bet_xy, alpha = [i for i in JNet.parameters()][-4:]
print([i for i in JNet.state_dict()][-4:])
print(torch.exp(bet_z), torch.exp(bet_xy), torch.exp(alpha), torch.exp(ez0))
JNet.eval()

dirpath = "_beads_roi_extracted_stackreg"
images = [os.path.join(dirpath, f) for f in sorted(os.listdir(dirpath))]
print(images)
losses = []
loss_fn = nn.MSELoss()
# for image_name in images[:-1]:
#     image_ = torch.load(image_name, map_location="cuda").to(torch.float32)
#     image = image_#(torch.clip(image_, min=0.1, max=1.) - 0.1) / (1.0 - 0.1)
#     outdict = JNet(image.to("cuda").unsqueeze(0))
#     output  = outdict["enhanced_image"]
#     output  = output.detach().cpu()
#     reconst = outdict["reconstruction"]
#     loss    = loss_fn(reconst, image).item()
#     qloss   = outdict["quantized_loss"]
#     est_params = outdict["blur_parameter"]
#     print("output ", torch.sum(output) * (0.05 * 0.05 * 0.05))
#     reconst = reconst.squeeze(0).detach().cpu().numpy()
#     losses.append(loss)
#     fig = plt.figure(figsize=(10, 10))
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     ax1.set_axis_off()
#     ax2.set_axis_off()
#     ax1.set_title('original image')
#     ax2.set_title(f'reconstruct image\n{model_name}')
#     plt.subplots_adjust(hspace=-0.0)
#     ax1.imshow(image_[0, :, i, :].to(device='cpu'),
#             cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
#     ax2.imshow(output[0, 0, :, i, :],
#             cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
#     plt.savefig(f'result/{model_name}_noclip_{image_name[30:-3]}.png', format='png', dpi=250)
# print(losses)

for image_name in images[:-1]:
    image_ = torch.load(image_name, map_location="cuda")
    c_, z_, x_, y_ = image_.shape
    ensenbled_output  = torch.zeros((c_, z_*scale, x_, y_), device='cpu')
    ensenbled_reconst = torch.zeros((c_, z_, x_, y_), device='cpu')
    ensenbled_loss    = 0
    ensenbled_qloss   = 0

    for _ in range(8):
        image   = sequentialflip(image_, i)#(torch.clip(image_, min=0.1, max=1.) - 0.1) / (1.0 - 0.1)
        outdict = JNet(image.to("cuda").unsqueeze(0))
        output  = outdict["enhanced_image"]
        output  = output.detach().cpu().squeeze(0)
        output  = sequentialflip(output.clone(), i)
        reconst = outdict["reconstruction"]
        reconst  = sequentialflip(reconst, i)
        image   = image.cpu()
        reconst = reconst.squeeze(0).detach().cpu()
        loss    = loss_fn(reconst, image).item()
        qloss   = outdict["quantized_loss"].item()
        ensenbled_output  += output  / 8
        ensenbled_reconst += reconst / 8
        ensenbled_loss    += loss  / 8
        ensenbled_qloss   += qloss / 8
    print("output ", torch.sum(ensenbled_output) * (0.05 * 0.05 * 0.05))
    reconst = reconst.squeeze(0).detach().cpu().numpy()
    losses.append(loss)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.set_title('original image')
    ax2.set_title(f'reconstruct image\n{model_name}')
    plt.subplots_adjust(hspace=-0.0)
    dirpath = "_beads_roi_extracted_stackreg/"
    #image_ = torch.load(image_name, map_location="cuda")
    image_ = torch.load(image_name, map_location="cuda")
    ax1.imshow(image_.cpu().numpy()[0,:,i, :], cmap="gray", vmin=0, vmax=1, aspect=10)
    ax2.imshow(ensenbled_output.cpu().numpy()[0, :, i, :],
             cmap="gray", vmin=0.0, vmax=1.0, aspect=1)
    plt.savefig(f'result/{model_name}_{image_name[30:-3]}.png', format='png', dpi=250)
print(losses)