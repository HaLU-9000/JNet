import torch
import torch.nn as nn
import numpy as np
from dataset import RandomCutDataset
import model

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

val_dataset   = RandomCutDataset(folderpath  =  'randomdata'     ,  ###
                                 imagename   =  '_x1'            ,
                                 labelname   =  '_label'         ,
                                 size        =  (768, 768, 768)  ,
                                 cropsize    =  (256,  64,  64)  ,
                                 I           =   20              ,
                                 low         =   16              ,
                                 high        =   20              ,
                                 scale       =    1              ,
                                 train       = False             ,
                                 seed        = 907               ,
                                )

model_name           = 'JNet_125_x1_softmax_temp_1e-6'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_list           = [(2, 1, 1)]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = (96, 192)
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  s_nblocks             = s_nblocks            ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_list            = scale_list           ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  ,
                  bet_xy                = 6.                   ,
                  bet_z                 = 35.                  ,
                  superres              = False                ,
                  )
JNet = JNet.to(device = device)
JNet.set_tau(0.1)
scale = 1

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'))
JNet.eval()

roc_list = []
n = 0
image, label = val_dataset[n]
output, _ = JNet(image.to(device).unsqueeze(0))
output    = output.detach().cpu().numpy()
if partial is not None:
    output = output[0, 0, partial[0]:partial[1]]
    label  = label[0, partial[0]:partial[1]]

for threshold in np.linspace(0, 1, 10):
    clipped_output    =  torch.tensor(output >= threshold)
    total     =  clipped_output.view(-1).shape[0]
    truepositive_rate   = torch.sum(clipped_output * label).item() / total
    falsepositive_rate  = torch.sum(clipped_output * (1 - label)).item() / total
    roc_list.append([falsepositive_rate, truepositive_rate])

roc_array = np.array(roc_list)
np.save(f'./roc_test/{model_name}_{n}.npy', roc_array)