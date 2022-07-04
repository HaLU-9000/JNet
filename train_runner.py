import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model as model
from dataset import PathDataset
from   train_loop import train_loop

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")
PathDataset
full_dataset = PathDataset(folderpath = 'datasetpath' ,
                           imagename  = '_x8'         ,    ###########
                           labelname  = '_label'      ,)
train_size           = int(len(full_dataset) * 0.8)
val_size             = len(full_dataset) - train_size
dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]                       ,
    generator = torch.Generator(device='cpu').manual_seed(701) , 
)
train_data  = DataLoader(dataset                                     ,
                         batch_size  = 2                             ,
                         shuffle     = True                          ,
                         pin_memory  = True                         ,
                         num_workers = os.cpu_count()                ,)
val_data    = DataLoader(val_dataset                                 ,
                         batch_size  = 2                             ,
                         shuffle     = False                         ,
                         pin_memory  = True                         ,
                         num_workers = os.cpu_count()                ,)

model_name              = 'JNet_54_x8' ################################
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
optimizer            = optim.Adam(JNet.parameters(), lr = 1e-6)
loss_fn              = nn.BCELoss()
train_loop(
    n_epochs     = 500       , ######################################
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