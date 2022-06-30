import sys
import os
sys.path.append('/home/haruhiko/Documents/JNet')
from numpy.random import seed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model as model
from dataloader import Blur, CustomDataset
from   train_loop import train_loop

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

full_dataset = torch.load('dataset/dataset128_x2_1.pt')
train_size           = int(len(full_dataset) * 0.8)
val_size             = len(full_dataset) - train_size
dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]                       ,
    generator = torch.Generator(device='cpu').manual_seed(419), 
)
g = torch.Generator(device = 'cpu')
g.manual_seed(621)
train_data  = DataLoader(dataset                                     ,
                         batch_size  = 1                             ,
                         shuffle     = True                          ,
                         generator   = g                             ,
                         pin_memory  = False                         ,
                         num_workers = 0                             ,
                         worker_init_fn = lambda x: seed(419)        ,)
val_data    = DataLoader(val_dataset                                 ,
                         batch_size  = 1                             ,
                         shuffle     = False                         ,
                         pin_memory  = False                         ,)

model_name = 'JNet_38_for_trainlooptest'
hidden_channels_list    = [1]#[16, 32, 64, 128, 256]
sr_hidden_channels_list = [1,1]#[64, 64]
nblocks                 = 2
activation              = nn.ReLU()
dropout                 = 0.5
torch.manual_seed(620)
JNet = model.JNet(hidden_channels_list    = hidden_channels_list    ,
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
JNet = JNet.to(device = device)
optimizer            = optim.Adam(JNet.parameters(), lr = 0.0005)
loss_fn              = nn.BCELoss()
train_loop(
    n_epochs     = 2         ,
    optimizer    = optimizer ,
    model        = JNet      ,
    loss_fn      = loss_fn   ,
    train_loader = train_data,
    val_loader   = val_data  ,
    device       = device    ,
    path         = 'model'   ,
    savefig_path = 'train'   ,
    model_name   = model_name,
    )

