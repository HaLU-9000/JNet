import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import RealDensityDataset
import model

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Inference on device {device}.")

scale    = 10
surround = False
surround_size = [32, 4, 4]
train_score     = torch.load('./beadsscore3/001_score.pt')

val_dataset   = RealDensityDataset(folderpath      =  'beadsdata2'      ,
                                   scorefolderpath =  'beadsscore2'     ,
                                   imagename       =  '001'            ,
                                   size            =  (1200, 512, 512) ,
                                   cropsize        =  ( 240, 112, 112) ,
                                   I               =  10               ,
                                   low             =   0               ,
                                   high            =   1               ,
                                   scale           =  scale            ,
                                   train           =  False            ,
                                   mask            =  False            ,
                                   surround        =  False            ,
                                   surround_size   =  [64, 8, 8]       ,
                                   seed            =  1204             ,
                                   score           =  train_score      ,
                                  )

model_name           = 'JNet_149_x10'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_factor         = (scale, 1, 1)
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres             = True if scale > 1 else False
reconstruct          = True
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_factor          = scale_factor         ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  ,
                  bet_xy                = 3.                   ,
                  bet_z                 = 17.5                 ,
                  superres              = superres             ,
                  reconstruct           = reconstruct          ,
                  )
JNet = JNet.to(device = device)
JNet.set_tau(1)
j = 120 // scale
i = 70

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()
for n in range(0,5):
    image, label= val_dataset[n+5]
    output, reconst= JNet(image.to("cuda").unsqueeze(0))
    output  = output.detach().cpu().numpy()
    reconst = reconst.squeeze(0).detach().cpu().numpy()
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
    ax1.set_title('plain\nreconstruct image')
    ax2.set_title('plain\noriginal image')
    ax3.set_title('plain\nsegmentation result')
    ax4.set_title('depth\nreconstruct')
    ax5.set_title('depth\noriginal image')
    ax6.set_title('depth\nsegmentation result')

    plt.subplots_adjust(hspace=-0.0)
    if partial is not None:
        ax1.imshow(reconst[0, partial[0]+j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax2.imshow(image[0, partial[0]+j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(output[0, 0, partial[0]+j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax4.imshow(reconst[0, partial[0]:partial[1], i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax5.imshow(image[0, partial[0]:partial[1], i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax6.imshow(output[0, 0, partial[0]:partial[1], i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)

    else:
        ax1.imshow(reconst[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax2.imshow(image[0, j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(output[0, 0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax4.imshow(reconst[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax5.imshow(image[0, :, i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax6.imshow(output[0, 0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)

    plt.savefig(f'result/{model_name}_result{n}.png', format='png', dpi=250)
    