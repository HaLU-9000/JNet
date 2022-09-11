import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model as model
from dataset import RandomCutDataset
from   train_loop import train_loop

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

train_dataset = RandomCutDataset(folderpath  =  'randomdata'     ,  ###
                                 imagename   =  '_x1'            ,
                                 labelname   =  '_label'         ,
                                 size        =  (768, 768, 768)  ,
                                 cropsize    =  (256,  64,  64)  ,
                                 I           =  200              ,
                                 low         =    0              ,
                                 high        =   16              ,
                                 scale       =    1              ,
                                )
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
train_data  = DataLoader(train_dataset                 ,
                         batch_size  = 1               ,
                         shuffle     = True            ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )
val_data    = DataLoader(val_dataset                   ,
                         batch_size  = 1               ,
                         shuffle     = False           ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )

model_name           = 'JNet_112_x1_partial'
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
JNet.load_state_dict(torch.load('model/JNet_83_x1_partial.pt'), strict=False)
optimizer            = optim.Adam(JNet.parameters(), lr = 1e-4)
scheduler            = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)
loss_fn              = nn.MSELoss()
midloss_fn           = nn.BCELoss()
train_loop(
    n_epochs     = 5000      , ####
    optimizer    = optimizer ,
    model        = JNet      ,
    loss_fn      = loss_fn   ,
    train_loader = train_data,
    val_loader   = val_data  ,
    device       = device    ,
    path         = 'model'   ,
    savefig_path = 'train'   ,
    model_name   = model_name,
    partial      = partial   ,
    scheduler    = scheduler ,
    es_patience  = 25        ,
    tau_init     = 1.        ,
    tau_lb       = 0.1       , 
    tau_sche     = 0.9999    ,
    reconstruct  = True      ,
    check_middle = True      ,
    midloss_fn   = midloss_fn,
    )