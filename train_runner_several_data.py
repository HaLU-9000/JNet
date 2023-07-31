import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model as model
from dataset import RandomCutDataset
from dataset import ParamScaler, Augmentation
from train_loop import train_loop_v2

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]

model_name           = 'JNet_257_bet_z_several'
hidden_channels_list = [16, 32, 64, 128, 256]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False
params               = {"mu_z"   : 0.2  , 
                        "sig_z"  : 0.2  , 
                        "log_bet_z"  :  np.log(20.).item() , 
                        "log_bet_xy" :  np.log( 1.).item() , 
                        "log_alpha"  :  np.log( 1.).item() , 
                        "sig_eps":  0.01,
                        "scale"  :  6
                        }

image_size = (1, 1, 240,  96,  96)
original_cropsize = [360, 120, 120]
param_estimation_list = [False, False, False, False, True]

JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  params                = params               ,
                  param_estimation_list = param_estimation_list,
                  superres              = superres             ,
                  reconstruct           = False                ,
                  apply_vq              = True                 ,
                  use_x_quantized       = False                ,
                  )
JNet = JNet.to(device = device)
#JNet.load_state_dict(torch.load('model/JNet_219_x6.pt'), strict=False)
params = [i for i in JNet.parameters()][:-4]
#params = JNet.parameters()

def warmup_func(epoch):
    return min(0.1 + 0.1 * epoch, 1.0)

optimizer            = optim.Adam(params, lr = 1e-3)
#scheduler            = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
scheduler            = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = warmup_func)
loss_fn              = nn.BCELoss()
midloss_fn           = nn.BCELoss()
param_loss_fn        = None

It = 200
Iv = 20

train_dataset0 = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_10' ,  ###
                                  imagename   =  f'_x{scale}'          , 
                                  labelname   =  '_label'              ,
                                  size        =  (1200, 500, 500)      ,
                                  cropsize    =  ( 240, 112, 112)      , 
                                  I             = It                  ,
                                  low           =   0                  ,
                                  high          =  16                  ,
                                  scale         =  scale               ,  ## scale
                                  mask          =  True                ,
                                  mask_size     =  [ 1, 10, 10]        ,
                                  mask_num      =  30                  ,  #( 1% of image)
                                  surround      =  surround            ,
                                  surround_size =  surround_size       ,
                                  )
val_dataset0  = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_10'   ,  ###
                                 imagename   =  f'_x{scale}'            ,     ## scale
                                 labelname   =  '_label'                ,
                                 size        =  (1200, 500, 500)        ,
                                 cropsize    =  ( 240, 112, 112)        ,
                                 I             =  Iv                    ,
                                 low           =  16                    ,
                                 high          =  20                    ,
                                 scale         =  scale                 ,   ## scale
                                 train         =  False                 ,
                                 mask          =  False                 ,
                                 surround      =  surround              ,
                                 surround_size =  surround_size         ,
                                 seed          =  907                   ,
                                )       

train_data0 = DataLoader(train_dataset0                ,
                         batch_size  = 1               ,
                         shuffle     = True            ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )
val_data0   = DataLoader(val_dataset0                  ,
                         batch_size  = 1               ,
                         shuffle     = False           ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )

train_dataset1 = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_20' ,  ###
                                  imagename   =  f'_x{scale}'          , 
                                  labelname   =  '_label'              ,
                                  size        =  (1200, 500, 500)      ,
                                  cropsize    =  ( 240, 112, 112)      , 
                                  I             = It                  ,
                                  low           =   0                  ,
                                  high          =  16                  ,
                                  scale         =  scale               ,  ## scale
                                  mask          =  True                ,
                                  mask_size     =  [ 1, 10, 10]        ,
                                  mask_num      =  30                  ,  #( 1% of image)
                                  surround      =  surround            ,
                                  surround_size =  surround_size       ,
                                  )
val_dataset1  = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_20'   ,  ###
                                 imagename   =  f'_x{scale}'            ,     ## scale
                                 labelname   =  '_label'                ,
                                 size        =  (1200, 500, 500)        ,
                                 cropsize    =  ( 240, 112, 112)        ,
                                 I             =  Iv                ,
                                 low           =  16                    ,
                                 high          =  20                    ,
                                 scale         =  scale                 ,   ## scale
                                 train         =  False                 ,
                                 mask          =  False                 ,
                                 surround      =  surround              ,
                                 surround_size =  surround_size         ,
                                 seed          =  907                   ,
                                )       

train_data1 = DataLoader(train_dataset1                ,
                         batch_size  = 1               ,
                         shuffle     = True            ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )
val_data1   = DataLoader(val_dataset1                  ,
                         batch_size  = 1               ,
                         shuffle     = False           ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )

train_dataset2 = RandomCutDataset(folderpath  =  '_var_num_beadsdata2' ,  ###
                                  imagename   =  f'_x{scale}'          , 
                                  labelname   =  '_label'              ,
                                  size        =  (1200, 500, 500)      ,
                                  cropsize    =  ( 240, 112, 112)      , 
                                  I             = It                  ,
                                  low           =   0                  ,
                                  high          =  16                  ,
                                  scale         =  scale               ,  ## scale
                                  mask          =  True                ,
                                  mask_size     =  [ 1, 10, 10]        ,
                                  mask_num      =  30                  ,  #( 1% of image)
                                  surround      =  surround            ,
                                  surround_size =  surround_size       ,
                                  )
val_dataset2  = RandomCutDataset(folderpath  =  '_var_num_beadsdata2'   ,  ###
                                 imagename   =  f'_x{scale}'            ,     ## scale
                                 labelname   =  '_label'                ,
                                 size        =  (1200, 500, 500)        ,
                                 cropsize    =  ( 240, 112, 112)        ,
                                 I             =  Iv                    ,
                                 low           =  16                    ,
                                 high          =  20                    ,
                                 scale         =  scale                 ,   ## scale
                                 train         =  False                 ,
                                 mask          =  False                 ,
                                 surround      =  surround              ,
                                 surround_size =  surround_size         ,
                                 seed          =  907                   ,
                                )       

train_data2 = DataLoader(train_dataset2                ,
                         batch_size  = 1               ,
                         shuffle     = True            ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )
val_data2   = DataLoader(val_dataset2                  ,
                         batch_size  = 1               ,
                         shuffle     = False           ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )

print(f"============= model {model_name} train started =============")

train_loop_v2(
              n_epochs         = 100                  , ####
              optimizer        = optimizer            ,
              model            = JNet                 ,
              loss_fn          = loss_fn              ,
              train_loader0    = train_data0          ,
              train_loader1    = train_data1          ,
              train_loader2    = train_data2          ,
              val_loader0      = val_data0            ,
              val_loader1      = val_data1            ,
              val_loader2      = val_data2            ,
              device           = device               ,
              path             = 'model'              ,
              savefig_path     = 'train'              ,
              model_name       = model_name           ,
              partial          = partial              ,
              scheduler        = scheduler            ,
              es_patience      = 15                   ,
              reconstruct      = False                ,
              check_middle     = False                ,
              midloss_fn       = midloss_fn           ,
              loss_weight      = 1                    ,
              qloss_weight     = 0                    ,
              verbose          = False                ,
              )