import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model_new as model
from dataset import RealDensityDataset, RandomCutDataset, RealSeveralDataset
from   train_loop import train_loop, ElasticWeightConsolidation

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 10
surround = False
surround_size = [32, 4, 4]
train_score   = torch.load('./_stackbeadsscore/002_score.pt') #torch.load('./sparsebeadslikescore/_x10_score.pt') #torch.load('./beadsscore/001_score.pt')
val_score     = torch.load('./_stackbeadsscore/002_score.pt') #None #torch.load('./sparsebeadslikescore/_x10_score.pt') #

train_dataset = RealDensityDataset(folderpath      =  '_stackbeadsdata'     ,
                                   scorefolderpath =  '_stackbeadsscore'    ,
                                   imagename       =  '002'            ,
                                   size            =  ( 650, 512, 512) , # size after segmentation
                                   cropsize        =  ( 240, 112, 112) , # size after segmentation
                                   I               = 200               ,
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
val_dataset   = RealDensityDataset(folderpath      =  '_stackbeadsdata'     ,
                                   scorefolderpath =  '_stackbeadsscore'    ,
                                   imagename       =  '002'            ,
                                   size            =  ( 650, 512, 512) , # size after segmentation
                                   cropsize        =  ( 240, 112, 112) ,
                                   I               =  20               ,
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

model_name           = 'JNet_295_realseveraldatasettest'
hidden_channels_list = [16, 32, 64, 128, 256]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False
params               = {"mu_z"       : 0.2               ,
                        "sig_z"      : 0.2               ,
                        "log_bet_z"  : np.log(30.).item(),
                        "log_bet_xy" : np.log(3.).item() ,
                        "log_alpha"  : np.log(1.).item() ,
                        "sig_eps": 0.01                  ,
                        "scale"  : 10                    ,
                        
                        }
reconstruct = True
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  params                = params               ,
                  superres              = superres             ,
                  reconstruct           = True                 ,
                  apply_vq              = True                 ,
                  use_x_quantized       = True                 ,
                  use_fftconv           = True                 ,
                  z                     = 161                  , 
                  x                     = 31                   , 
                  y                     = 31                   ,
                  )
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load('model/JNet_265_vibration.pt'),
                     strict=False)
init_log_ez0 = (torch.tensor(params["mu_z"]) + 0.5 \
                * torch.tensor(params["sig_z"]) ** 2).to(device)
JNet.image.emission.log_ez0.data = init_log_ez0
JNet.image.blur.log_bet_xy.data  = torch.tensor(params["log_bet_xy"]).to(device)
JNet.image.blur.log_bet_z.data   = torch.tensor(params["log_bet_z"]).to(device)
#print([i for i in JNet.parameters()][-4:])

params = JNet.parameters()

optimizer            = optim.Adam(params, lr = 1e-3)
scheduler            = None #= optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
loss_fn              = nn.MSELoss()
midloss_fn           = nn.BCELoss()

# ewc_dataset   = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_30_hill' ,  ###
#                                  imagename   =  f'_x6'                , 
#                                  labelname   =  '_label'              ,
#                                  size        =  (1200, 500, 500)      ,
#                                  cropsize    =  ( 240, 112, 112)      ,
#                                  I             = 800                  ,
#                                  low           =   0                  ,
#                                  high          =  16                  ,
#                                  scale         =  scale               ,  ## scale
#                                  mask          =  True                ,
#                                  mask_size     =  [ 1, 10, 10]        ,
#                                  mask_num      =  10                  ,  #( 1% of image)
#                                  surround      =  surround            ,
#                                  surround_size =  surround_size       ,
#                                  )

# ewc_data    = DataLoader(ewc_dataset                   ,
#                          batch_size  = 1               ,
#                          shuffle     = True            ,
#                          pin_memory  = True            ,
#                          num_workers = os.cpu_count()  ,
#                          )
# ewc = ElasticWeightConsolidation(model           = JNet,
#                                  prev_dataloader = ewc_data,
#                                  loss_fn         = loss_fn,
#                                  init_num_batch  = 100,
#                                  is_vibrate      = True,
#                                  device          = device,
#                                  skip_register   = True   )
# torch.save(JNet.state_dict(), f'model/JNet_265_vibration.pt')

print(f"============= model {model_name} train started =============")
train_loop(
    n_epochs         = 100         , ####
    optimizer        = optimizer   ,
    model            = JNet        ,
    loss_fn          = loss_fn     ,
    param_loss_fn    = None        ,
    train_loader     = train_data  ,
    val_loader       = val_data    ,
    device           = device      ,
    path             = 'model'     ,
    savefig_path     = 'train'     ,
    model_name       = model_name  ,
    param_normalize  = None        ,
    augment          = None        ,
    val_augment      = None        ,
    ewc              = None        ,
    partial          = partial     ,
    scheduler        = scheduler   ,
    es_patience      = 20          ,
    reconstruct      = reconstruct ,
    check_middle     = False       ,
    midloss_fn       = midloss_fn  ,
    is_vibrate       = True        ,
    loss_weight      = 1           ,
    qloss_weight     = 1           ,
    paramloss_weight = 0           ,
    verbose          = False       ,
    )
