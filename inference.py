import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import PathDataset
import model

model_name = 'JNet_54_x8'
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

full_dataset = PathDataset(folderpath = 'datasetpath' ,
                           imagename  = '_x8'         ,    ###########
                           labelname  = '_label'      ,)
train_size           = int(len(full_dataset) * 0.8)
val_size             = len(full_dataset) - train_size
dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]                       ,
    generator = torch.Generator(device='cpu').manual_seed(701) , 
)
g = torch.Generator(device = 'cpu')
g.manual_seed(621)

hidden_channels_list = [16, 32, 64, 128, 256]
scale_list           = [(2, 1, 1), (2, 1, 1), (2, 1, 1)]
nblocks              = 2
activation           = nn.ReLU()
dropout              = 0.5
torch.manual_seed(703)
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_list            = scale_list           ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  ,
                  bet_xy                = 6.                   ,
                  bet_z                 = 35.                  ,)
JNet = JNet.to(device = device)

j = 0
i = 0
scale = 8

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'))
JNet.eval()
for n in range(0,5):
    output = JNet(val_dataset[n][0].to("cuda").unsqueeze(0))[0].detach().cpu().numpy()
    fig = plt.figure(figsize=(20, 15))
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

    ax3.imshow(val_dataset[n][1][0, j, :, :].to(device='cpu'),
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    ax1.imshow(val_dataset[n][0][0, j, :, :].to(device='cpu'),
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    ax2.imshow(output[0, 0, j, :, :],
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    ax6.imshow(val_dataset[n][1][0, :, i, :].to(device='cpu'),
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    ax4.imshow(val_dataset[n][0][0, :, i, :].to(device='cpu'),
            cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
    ax5.imshow(output[0, 0, :, i, :],
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    plt.savefig(f'result/{model_name}_result{n}.png', format='png', dpi=250)