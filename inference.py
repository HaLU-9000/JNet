import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import RandomCutDataset
import model

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

val_dataset   = RandomCutDataset(folderpath    =  'beadslikedata2' , 
                                 imagename     =  '_x1'            ,
                                 labelname     =  '_label'         ,
                                 size          =  (1200, 500, 500) ,
                                 cropsize      =  ( 240, 112, 112) ,
                                 I             =  10               ,
                                 low           =  16               ,
                                 high          =  20               ,
                                 scale         =  1                ,
                                 train         =  False            ,
                                 mask          =  False            ,
                                 surround      =  True             ,
                                 surround_size =  [16, 4, 4]       ,
                                 seed          =  907              ,
                                )

model_name           = 'JNet_147_x1'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_factor         = (1, 1, 1)
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_factor          = scale_factor         ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  ,
                  bet_xy                = 6.                   ,
                  bet_z                 = 35.                  ,
                  superres              = False                ,
                  reconstruct           = True                 ,
                  )
JNet = JNet.to(device = device)
JNet.set_tau(1)
j = 60
i = 60
scale = scale_factor[0]

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()
for n in range(0,5):
    image, label= val_dataset[n]
    output, reconst= JNet(image.to("cuda").unsqueeze(0))
    output  = output.detach().cpu().numpy()
    reconst = reconst.squeeze(0).detach().cpu().numpy()
    fig = plt.figure(figsize=(25, 15))
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    ax5.set_axis_off()
    ax6.set_axis_off()
    ax7.set_axis_off()
    ax8.set_axis_off()
    ax1.set_title('plain\nreconstruct image')
    ax2.set_title('plain\noriginal image')
    ax3.set_title('plain\nsegmentation result')
    ax4.set_title('plain\nlabel')
    ax5.set_title('depth\nreconstruct')
    ax6.set_title('depth\noriginal image')
    ax7.set_title('depth\nsegmentation result')
    ax8.set_title('depth\nlabel')
    plt.subplots_adjust(hspace=-0.0)
    if partial is not None:
        ax1.imshow(reconst[0, partial[0]+j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax2.imshow(image[0, partial[0]+j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(output[0, 0, partial[0]+j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax4.imshow(label[0, partial[0]+j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax5.imshow(reconst[0, partial[0]:partial[1], i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax6.imshow(image[0, partial[0]:partial[1], i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax7.imshow(output[0, 0, partial[0]:partial[1], i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax8.imshow(label[0, partial[0]:partial[1], i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    else:
        ax1.imshow(reconst[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax2.imshow(image[0, j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(output[0, 0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax4.imshow(label[0, j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax5.imshow(reconst[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax6.imshow(image[0, :, i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax7.imshow(output[0, 0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax8.imshow(label[0, :, i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    plt.savefig(f'result/{model_name}_result{n}.png', format='png', dpi=250)
    