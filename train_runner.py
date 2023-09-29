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
model_name = 'JNet_342_gaussianpsf_pretrain' # 318_pretrain_b1
params     = {"hidden_channels_list"  : [4, 8, 16, 32, 64]                ,
              "attn_list"             : [False, False, False, False, True],     
              "nblocks"               : 2                                 ,     
              "activation"            : nn.ReLU(inplace=True)             ,     
              "dropout"               : 0.5                               ,     
              "superres"              : True                              ,     
              "partial"               : None                              ,
              "reconstruct"           : False                             ,     
              "apply_vq"              : False                             ,     
              "use_fftconv"           : True                              ,     
              "use_x_quantized"       : False                             ,     
              "mu_z"                  : 0.1                               ,
              "sig_z"                 : 0.1                               ,
              "blur_mode"             : "gaussian"                        ,
              "size_x"                : 51                                ,
              "size_y"                : 51                                ,
              "size_z"                : 161                               ,
              "NA"                    : 0.80                              , # # # # param # # # #
              "wavelength"            : 0.910                             , # microns # # # # param # # # #
              "M"                     : 25                                , # magnification # # # # param # # # #
              "ns"                    : 1.4                               , # specimen refractive index (RI)
              "ng0"                   : 1.5                               , # coverslip RI design value
              "ng"                    : 1.5                               , # coverslip RI experimental value
              "ni0"                   : 1.5                               , # immersion medium RI design value
              "ni"                    : 1.5                               , # immersion medium RI experimental value
              "ti0"                   : 150                               , # microns, working distance (immersion medium thickness) design value
              "tg0"                   : 170                               , # microns, coverslip thickness design value
              "tg"                    : 170                               , # microns, coverslip thickness experimental value
              "res_lateral"           : 0.05                              , # microns # # # # param # # # #
              "res_axial"             : 0.05                              , # microns # # # # param # # # #
              "pZ"                    : 0                                 , # microns, particle distance from coverslip
              "bet_z"                 : 30.                               ,
              "bet_xy"                :  3.                               ,
              "sig_eps"               : 0.01                              ,
              "scale"                 : 10                                ,
              "device"                : device                            ,
              }

JNet = model.JNet(params)
JNet = JNet.to(device = device)
#JNet.load_state_dict(torch.load('model/JNet_219_x6.pt'), strict=False)
train_params = JNet.parameters()

def warmup_func(epoch):
    return min(0.1 + 0.1 * epoch, 1.0)

optimizer            = optim.Adam(train_params, lr = 1e-3)
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
train_loop(n_epochs         = 200                  , ####
           optimizer        = optimizer            ,
           model            = JNet                 ,
           loss_fn          = loss_fn              ,
           train_loader     = train_data           ,
           val_loader       = val_data             ,
           device           = device               ,
           path             = model_path           ,
           savefig_path     = 'train'              ,
           model_name       = model_name           ,
           partial          = params["partial"]    ,
           ewc              = None                 ,
           params           = params               ,
           scheduler        = scheduler            ,
           es_patience      = 20                   ,
           reconstruct      = False                ,
           check_middle     = False                ,
           midloss_fn       = midloss_fn           ,
           is_instantblur   = True                 ,
           is_vibrate       = True                 ,
           loss_weight      = 1                    ,
           qloss_weight     = 0                    ,
           )