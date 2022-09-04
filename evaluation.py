import torch
import torch.nn as nn

import model
from dataset import PathDataset ## cropped dataset
from metrics import jiffs

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

model_name           = 'JNet_77_x1'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_list           = [(2, 1, 1)]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU()
dropout              = 0.5
partial              = (64, 192) ########################
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
scale = 1

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'))
JNet.eval()

val_dataset   = PathDataset(folderpath  =  'datasetpath'     ,  ###
                            imagename   =  '_x1'            ,
                            labelname   =  '_label'         ,
                            )