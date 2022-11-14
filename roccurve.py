import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
j = 60
i = 30
scale = 1

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'))
JNet.eval()

for n in range(0,1):
    image, label= val_dataset[n]
    output, _ = JNet(image.to(device).unsqueeze(0))
    output    = output.detach().cpu().numpy()
    