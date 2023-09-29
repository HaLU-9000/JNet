import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model_new as model
from dataset import RealDensityDataset, RandomCutDataset
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
                                   size            =   (650, 512, 512) , # size after segmentation
                                   cropsize        =  ( 240, 112, 112) , # size after segmentation
                                   I               =  200               ,
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
                                   I               =  20              ,
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

model_name            = 'JNet_338_physics'
pretrained_model_name = 'JNet_326_1_4_cross_attn_1'

params     = {"hidden_channels_list" : [4, 8, 16, 32, 64]                 ,
              "attn_list"            : [False, False, False, False, True] ,     
              "nblocks"              : 2                                  ,     
              "activation "          : nn.ReLU(inplace=True)              ,     
              "dropout"              : 0.5                                ,     
              "superres"             : True                               ,     
              "reconstruct"          : True                               ,     
              "apply_vq"             : True                               ,     
              "use_fftconv"          : True                               ,     
              "use_x_quantized"      : True                               ,     
              "mu_z"                 : 0.1                                ,
              "sig_z"                : 0.1                                ,
              "size_x"               : 51                                 ,
              "size_y"               : 51                                 ,
              "size_z"               : 161                                ,
              "NA"                   : 1.33                               ,
              "wavelength"           : 0.910                              ,
              "M"                    : 25                                 ,
              "res_lateral"          : 0.05                               ,
              "res_axial"            : 0.5                                ,
              "sig_eps"              : 0.01                               ,
              "scale"                : 10                                 ,
              }

JNet = model.JNet(params)
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{pretrained_model_name}.pt'),
                     strict=False)
#print([i for i in JNet.parameters()][-4:])

params = JNet.parameters()

optimizer            = optim.Adam(params, lr = 1e-3)
scheduler            = None #= optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
loss_fn              = nn.MSELoss()
midloss_fn           = nn.BCELoss()

ewc_dataset   = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_30_fft_blur' ,  ###
                                 imagename   =  f'_x6'                , 
                                 labelname   =  '_label'              ,
                                 size        =  (1200, 500, 500)      ,
                                 cropsize    =  ( 240, 112, 112)      , 
                                 I             = 800                  ,
                                 low           =   0                  ,
                                 high          =  16                  ,
                                 scale         =  scale               ,  ## scale
                                 mask          =  True                ,
                                 mask_size     =  [ 1, 10, 10]        ,
                                 mask_num      =  30                  ,  #( 1% of image)
                                 surround      =  surround            ,
                                 surround_size =  surround_size       ,
                                 )

ewc_data    = DataLoader(ewc_dataset                   ,
                         batch_size  = 1               ,
                         shuffle     = True            ,
                         pin_memory  = True            ,
                         num_workers = os.cpu_count()  ,
                         )
#ewc = ElasticWeightConsolidation(model           = JNet,
#                                 prev_dataloader = ewc_data,
#                                 loss_fn         = loss_fn,
#                                 init_num_batch  = 100,
#                                 is_vibrate      = True,
#                                 device          = device,
#                                 skip_register   = False  )
#torch.save(JNet.state_dict(), f'model/JNet_265_vibration.pt')
ewc = None
print(f"============= model {model_name} train started =============")
train_loop(
    n_epochs         = 500         , ####
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
    ewc              = ewc         ,
    partial          = params["partial"]     ,
    scheduler        = scheduler   ,
    es_patience      = 20          ,
    reconstruct      = params["reconstruct"] ,
    check_middle     = False       ,
    midloss_fn       = midloss_fn  ,
    is_vibrate       = True        ,
    loss_weight      = 1           ,
    qloss_weight     = 1           ,
    paramloss_weight = 0           ,
    verbose          = False       ,
    )
