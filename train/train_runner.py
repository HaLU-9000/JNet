import torch
import torch.nn as nn

import model.models as model

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")


hidden_channels_list    = [16, 32, 64, 128, 256]
sr_hidden_channels_list = [64, 64]
nblocks                 = 2
activation              = nn.ReLU()
dropout                 = 0.5
torch.manual_seed(620)

model = model.JNet(hidden_channels_list    = hidden_channels_list    ,
                   nblocks                 = nblocks                 ,
                   activation              = activation              ,
                   dropout                 = dropout                 ,
                   sr_hidden_channels_list = sr_hidden_channels_list ,
                   sr_out_channels         = 8                       ,
                   scale_factor            = 2                       ,
                   scale                   = 2                       ,
                   mu_z                    = 0.2                     ,
                   sig_z                   = 0.2                     ,
                   bet_xy                  = 6.                      ,
                   bet_z                   = 35.                     ,)
model = model.to(device = device)

