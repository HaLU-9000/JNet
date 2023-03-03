import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import RealDensityDataset
import model

vis_mseloss = False

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Inference on device {device}.")

scale    = 10
surround = False
surround_size = [32, 4, 4]
val_score     = torch.load('./spinescore0/020_score.pt') 
val_dataset   = RealDensityDataset(folderpath      =  'spinedata0'     ,
                                   scorefolderpath =  'spinescore0'    ,
                                   imagename       =  '020'            ,
                                   size            =  ( 282, 512, 512) , # size after segmentation
                                   cropsize        =  ( 240, 112, 112) ,
                                   I               =  10               ,
                                   low             =   0               ,
                                   high            =   1               ,
                                   scale           =  scale            ,
                                   train           =  False            ,
                                   mask            =  False            ,
                                   surround        =  False            ,
                                   surround_size   =  surround_size    ,
                                   seed            =  1204             ,
                                   score           =  val_score        ,
                                  )

val_dataset_  = RealDensityDataset(folderpath      =  'spinerawdata0'  ,
                                   scorefolderpath =  'spinescore0'    ,
                                   imagename       =  '020'            ,
                                   size            =  ( 282, 512, 512) , # size after segmentation
                                   cropsize        =  ( 240, 112, 112) ,
                                   I               =  10               ,
                                   low             =   0               ,
                                   high            =   1               ,
                                   scale           =  scale            ,
                                   train           =  False            ,
                                   mask            =  False            ,
                                   surround        =  False            ,
                                   surround_size   =  surround_size    ,
                                   seed            =  1204             ,
                                   score           =  val_score        ,
                                  )

model_name           = 'JNet_179_x6'
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
                  bet_xy                = 4.43864              ,
                  bet_z                 = 27.7052              ,
                  alpha                 = 74.9664              ,
                  superres              = superres             ,
                  reconstruct           = reconstruct          ,
                  )
JNet = JNet.to(device = device)
JNet.set_tau(1)
j   = 12
j_s = j // scale
i = 70

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()

if vis_mseloss == False:
    n = 5
    if n == 5: #for n in range(0,10):
        image , label = val_dataset[n]
        image_, label = val_dataset_[n]
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
        ax1.set_title(f'plain\nreconstruct image\n{model_name}')
        ax2.set_title('plain\noriginal image')
        ax3.set_title(f'plain\nsegmentation result\n{model_name}')
        ax4.set_title(f'depth\nreconstruct\n{model_name}')
        ax5.set_title('depth\noriginal image')
        ax6.set_title(f'depth\nsegmentation result\n{model_name}')
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
            ax1.imshow(reconst[0, j_s, :, :],
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax2.imshow(image_[0, j_s, :, :].to(device='cpu'),
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax3.imshow(output[0, 0, j, :, :],
                    cmap='gray', vmin=0.2, vmax=1.0, aspect=1)
            ax4.imshow(reconst[0, :, i, :],
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
            ax5.imshow(image_[0, :, i, :].to(device='cpu'),
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
            ax6.imshow(output[0, 0, :, i, :],
                    cmap='gray', vmin=0.2, vmax=1.0, aspect=1)

        plt.savefig(f'result/{model_name}_realresult{n}_raw.png', format='png', dpi=250)

else:
    for n in range(0, 10):
        #if n == 7 or n==9:
            image, label= val_dataset[n]
            output, reconst= JNet(image.to("cuda").unsqueeze(0))
            output  = output.detach().cpu().numpy()
            reconst = reconst.squeeze(0).detach().cpu().numpy()
            mse = (image - reconst) ** 2
            print(n, " image max min",image.max(), image.min())
            print(" reconst max min", reconst.max(), reconst.min())

            msemax = mse.max()
            msemean= mse.mean()
            vmax = 0.25
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
            ax1.set_title(f'plain\nreconstruct image\n{model_name}\nmax {reconst.max()}\nmin {reconst.min()}\nmean {reconst.mean()}')
            ax2.set_title(f'plain\noriginal image\nmax {image.max()}\nmin {image.min()}\nmean {image.mean()}')
            ax3.set_title(f'plain\nmseloss\nmax mse={mse.max()}\nmin mse={mse.min()}\nmean mse={mse.mean()}')
            ax4.set_title('depth')
            ax5.set_title('depth')
            ax6.set_title('depth')
            plt.subplots_adjust(hspace=0.1)
            ax1.imshow(reconst[0, j_s, :, :],
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax2.imshow(image[0, j_s, :, :].to(device='cpu'),
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax3.imshow(mse[0, j_s, :, :], vmin=0.0, vmax=vmax, aspect=1)
            ax4.imshow(reconst[0, :, i, :],
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
            ax5.imshow(image[0, :, i, :].to(device='cpu'),
                    cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
            ax6.imshow(mse[0, :, i, :], vmin=0.0, vmax=vmax, aspect=scale)
            plt.savefig(f'result/{model_name}_mseloss_{n}_0_05.png', format='png', dpi=250)