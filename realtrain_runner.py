import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model as model
from dataset import RealDensityDataset
from   train_loop import train_loop

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 10
surround = False
surround_size = [32, 4, 4]
train_score   = torch.load('./sparsebeadslikescore/_x10_score.pt') #torch.load('./beadsscore/001_score.pt')
val_score     = torch.load('./sparsebeadslikescore/_x10_score.pt') #torch.load('./beadsscore/002_score.pt')

train_dataset = RealDensityDataset(folderpath      =  'sparsebeadslikedata' ,
                                   scorefolderpath =  'sparsebeadslikescore',
                                   imagename       =  '_x10'            ,
                                   size            =  (1200, 500, 500) , # size after segmentation
                                   cropsize        =  ( 240, 112, 112) , # size after segmentation
                                   I               =  200              ,
                                   low             =   0               ,
                                   high            =   1               ,
                                   scale           =  scale            ,   ## scale
                                   train           =  True             ,
                                   mask            =  True             ,
                                   mask_num        =  10               ,
                                   mask_size       =  [1, 10, 10]      ,
                                   surround        =  surround         ,
                                   surround_size   =  surround_size    ,
                                   score           =  train_score      ,
                                  )
val_dataset   = RealDensityDataset(folderpath      =  'sparsebeadslikedata' ,
                                   scorefolderpath =  'sparsebeadslikescore',
                                   imagename       =  '_x10'            ,
                                   size            =  (1200, 500, 500) ,
                                   cropsize        =  ( 240, 112, 112) ,
                                   I               =  10               ,
                                   low             =   0               ,
                                   high            =   1               ,
                                   scale           =  scale            ,
                                   train           =  False            ,
                                   mask            =  False            ,
                                   surround        =  False            ,
                                   surround_size   =  [64, 8, 8]       ,
                                   seed            =  1204             ,
                                   score           =  val_score        ,
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

model_name           = 'JNet_161_x10'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_factor         = (scale, 1, 1)
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres             = True if scale > 1 else False
reconstruct          = True
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_factor          = scale_factor         ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  ,
                  bet_xy                = 3.                   ,
                  bet_z                 = 17.5                 ,
                  superres              = superres             ,
                  reconstruct           = reconstruct          ,
                  )
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load('model/JNet_149_x10.pt'), strict=False)
#params = [i for i in JNet.parameters()][:-4]
params = JNet.parameters()
optimizer            = optim.Adam(params, lr = 1e-5)
scheduler            = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
loss_fn              = nn.MSELoss()
midloss_fn           = nn.BCELoss()
print(f"============= model {model_name} train started =============")
train_loop(
    n_epochs     = 500         , ####
    optimizer    = optimizer   ,
    model        = JNet        ,
    loss_fn      = loss_fn     ,
    train_loader = train_data  ,
    val_loader   = val_data    ,
    device       = device      ,
    path         = 'model'     ,
    savefig_path = 'train'     ,
    model_name   = model_name  ,
    partial      = partial     ,
    scheduler    = scheduler   ,
    es_patience  = 15          ,
    tau_init     = 1           ,
    tau_lb       = 0.1         , 
    tau_sche     = 0.9999      ,
    reconstruct  = reconstruct ,
    check_middle = False       ,
    midloss_fn   = midloss_fn  ,
    )