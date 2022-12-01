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

train_dataset = RandomCutDataset(folderpath    =  'beadslikedata'  ,
                                 imagename     =  '_x1'            ,
                                 labelname     =  '_label'         ,
                                 size          =  (1200, 500, 500) ,
                                 cropsize      =  ( 240, 112, 112) , 
                                 I             = 200               ,
                                 low           =   0               ,
                                 high          =  16               ,
                                 scale         =   1               ,
                                 mask          =  True             ,
                                 mask_size     =  [10, 10, 10]     ,
                                 mask_num      =  1                ,
                                 surround      =  True             ,
                                 surround_size =  [32, 8, 8]       ,
                                 )
val_dataset   = RandomCutDataset(folderpath    =  'beadslikedata'  , 
                                 imagename     =  '_x1'            ,
                                 labelname     =  '_label'         ,
                                 size          =  (1200, 500, 500) ,
                                 cropsize      =  ( 240, 112, 112) ,
                                 I             =  10               ,
                                 low           =  16               ,
                                 high          =  20               ,
                                 scale         =   1               ,
                                 train         =  False            ,
                                 mask          =  False            ,
                                 surround      =  True             ,
                                 surround_size =  [32, 8, 8]       ,
                                 seed          =  907              ,
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

model_name           = 'JNet_140_x1'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_factor         = (1, 1, 1)
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_factor          = scale_factor         ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  ,
                  bet_xy                = 6.                   ,
                  bet_z                 = 35.                  ,
                  superres              = True                 ,
                  )
JNet = JNet.to(device = device)
#JNet.load_state_dict(torch.load('model/JNet_83_x1_partial.pt'), strict=False)
params = [i for i in JNet.parameters()][:-4]
#params = JNet.parameters()
optimizer            = optim.Adam(params, lr = 1e-4)
scheduler            = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)
loss_fn              = nn.BCELoss()
midloss_fn           = nn.BCELoss()
print(f"============= model {model_name} train started =============")
train_loop(
    n_epochs     = 500        , ####
    optimizer    = optimizer  ,
    model        = JNet       ,
    loss_fn      = loss_fn    ,
    train_loader = train_data ,
    val_loader   = val_data   ,
    device       = device     ,
    path         = 'model'    ,
    savefig_path = 'train'    ,
    model_name   = model_name ,
    partial      = partial    ,
    scheduler    = scheduler  ,
    es_patience  = 25         ,
    tau_init     = 1.         ,
    tau_lb       = 1          , 
    tau_sche     = 1          ,
    reconstruct  = False      ,
    check_middle = False      ,
    midloss_fn   = midloss_fn ,
    )