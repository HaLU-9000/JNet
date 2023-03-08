import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import RandomCutDataset
import model

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]

val_dataset   = RandomCutDataset(folderpath  =  'spinelikedata0'   ,  ###
                                 imagename   =  f'_x{scale}'       ,     ## scale
                                 labelname   =  '_label'           ,
                                 size        =  (1200, 500, 500)   ,
                                 cropsize    =  ( 240, 112, 112)  ,
                                 I             =  10               ,
                                 low           =  16               ,
                                 high          =  20               ,
                                 scale         =  scale            ,   ## scale
                                 train         =  False            ,
                                 mask          =  False            ,
                                 mask_num      =  100              ,
                                 mask_size     =  [1, 10, 10]      ,
                                 surround      =  surround         ,
                                 surround_size =  surround_size    ,
                                 seed          =  907              ,
                                )

model_name           = 'JNet_178_x6'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_factor         = (scale, 1, 1)
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_factor          = scale_factor         ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  , 
                  bet_z                 = 23.5329              ,
                  bet_xy                = 1.00000              ,
                  alpha                 = 0.9544               ,
                  superres              = superres             ,
                  reconstruct           = True                 ,
                  )
JNet = JNet.to(device = device)
JNet.set_tau(1)
j = 120
i = 60
j_s = j // scale

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()
for n in range(0,1):
    image, label= val_dataset[n]
    torch.save(image, f'./result_psj/{model_name}_image{n}.pt')
    torch.save(label, f'./result_psj/{model_name}_label{n}.pt')
    output, reconst= JNet(image.to("cuda").unsqueeze(0))
    output  = output.detach().cpu().numpy()
    torch.save(output, f'./result_psj/{model_name}_result{n}.pt')
    reconst = reconst.squeeze(0).detach().cpu().numpy()

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
    if partial is not None:
        ax1.imshow(reconst[0, partial[0]+j_s, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax2.imshow(image[0, partial[0]+j_s, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(output[0, 0, partial[0]+j_s, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax4.imshow(label[0, partial[0]+j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax5.imshow(reconst[0, partial[0]:partial[1], i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax6.imshow(image[0, partial[0]:partial[1], i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        #ax7.imshow(output[0, 0, partial[0]:partial[1], i, :],
        #        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        #ax8.imshow(label[0, partial[0]:partial[1], i, :].to(device='cpu'),
        #        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    else:
        #ax1.imshow(reconst[0, j_s, :, :],
        #        cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax1.imshow(image[0, j_s, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, aspect=1)
        ax2.imshow(output[0, 0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax3.imshow(label[0, j, :, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        #ax5.imshow(reconst[0, :, i, :],
        #        cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
        ax4.imshow(image[0, :, i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, aspect=scale)
        ax5.imshow(output[0, 0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
        ax6.imshow(label[0, :, i, :].to(device='cpu'),
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    plt.savefig(f'result_psj/{model_name}_sim_result{n}.png', format='png', dpi=250)
    