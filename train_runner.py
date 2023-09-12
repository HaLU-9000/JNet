import os
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model_new as model
from dataset import RandomCutDataset
from dataset import ParamScaler, Augmentation
from train_loop import train_loop, ElasticWeightConsolidation

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]
model_name           = 'JNet_322_quorter_channel' # 318_pretrain_b1
hidden_channels_list = [4, 8, 16, 32, 64]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False
params               = {"mu_z"   : 0.2  , 
                        "sig_z"  : 0.2  , 
                        "log_bet_z"  :  20. , 
                        "log_bet_xy" :   1. , 
                        "log_alpha"  :   1. ,
                        "log_k"      :   0. ,
                        "log_l"      :   0. ,
                        "sig_eps":  0.01,
                        "scale"  :  6
                        }

image_size = (1, 1, 240,  112,  112)
original_cropsize = [360, 120, 120]
param_estimation_list = [False, False, False, False, True]

JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  params                = params               ,
                  superres              = superres             ,
                  reconstruct           = False                ,
                  apply_vq              = True                 ,
                  use_x_quantized       = False             
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

train_dataset = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_30_fft_blur' ,  ###
                                 imagename   =  f'_x{scale}'          , 
                                 labelname   =  '_label'              ,
                                 size        =  (1200, 500, 500)      ,
                                 cropsize    =  ( 240, 112, 112)      , 
                                 I             = 200                  ,
                                 low           =   0                  ,
                                 high          =  16                  ,
                                 scale         =  scale               ,  ## scale
                                 mask          =  True                ,
                                 mask_size     =  [ 1, 10, 10]        ,
                                 mask_num      =  30                  ,  #( 1% of image)
                                 surround      =  surround            ,
                                 surround_size =  surround_size       ,
                                 )
val_dataset   = RandomCutDataset(folderpath  =  '_var_num_beadsdata2_30_fft_blur'   ,  ###
                                 imagename   =  f'_x{scale}'            ,     ## scale
                                 labelname   =  '_label'                ,
                                 size        =  (1200, 500, 500)        ,
                                 cropsize    =  ( 240, 112, 112)        ,
                                 I             =  20                    ,
                                 low           =  16                    ,
                                 high          =  20                    ,
                                 scale         =  scale                 ,   ## scale
                                 train         =  False                 ,
                                 mask          =  False                 ,
                                 surround      =  surround              ,
                                 surround_size =  surround_size         ,
                                 seed          =  907                   ,
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

print(f"============= model {model_name} train started =============")
model_path = 'model'
train_loop(
           n_epochs         = 200                  , ####
           optimizer        = optimizer            ,
           model            = JNet                 ,
           loss_fn          = loss_fn              ,
           param_loss_fn    = param_loss_fn        ,
           train_loader     = train_data           ,
           val_loader       = val_data             ,
           device           = device               ,
           path             = model_path           ,
           savefig_path     = 'train'              ,
           model_name       = model_name           ,
           param_normalize  = None                 ,
           augment          = None                 ,
           val_augment      = None                 ,
           partial          = partial              ,
           ewc              = None                 ,
           scheduler        = scheduler            ,
           es_patience      = 20                   ,
           reconstruct      = False                ,
           check_middle     = False                ,
           midloss_fn       = midloss_fn           ,
           is_randomblur    = False                ,
           is_vibrate       = True                 ,
           loss_weight      = 1                    ,
           qloss_weight     = 0                    ,
           paramloss_weight = 0                    ,
           verbose          = False                ,
           )

#JNet.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'), strict=False)
#ElasticWeightConsolidation(model           = JNet,
#                           prev_dataloader = train_data,
#                           loss_fn         = loss_fn,
#                           init_num_batch  = 100,
#                           is_vibrate      = True,
#                           device          = device)
#torch.save(JNet.state_dict(), f'{model_path}/{model_name}.pt')