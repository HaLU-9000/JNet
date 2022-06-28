import torch
import torch.nn as nn 
from utils import ModelSizeEstimator
import model
model_name = 'JNet_41'
hidden_channels_list    = [16, 32, 64, 128, 256, 512]
sr_hidden_channels_list = [64, 64]
nblocks                 = 2
activation              = nn.ReLU()
dropout                 = 0.5
torch.manual_seed(620)
JNet = model.JNet(hidden_channels_list     = hidden_channels_list    ,
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
ModelSizeEstimator(JNet).__call__()