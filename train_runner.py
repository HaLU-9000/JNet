import os
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model as model
from dataset import LabelandBlurParamsDataset, RandomBlurbyModelDataset, gen_imaging_parameters
from dataset import ParamScaler, Augmentation
from train_loop import train_loop

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]

params_ranges = {"mu_z"   : [0,   1, 0.2   , 0.01 ],
                 "sig_z"  : [0,   1, 0.2   , 0.01 ],
                 "bet_z"  : [0 , 25,  20   , 2.   ],
                 "bet_xy" : [0,   2,   1.  , 1.   ],
                 "alpha"  : [0,   2,   1.  , 0.01 ],
                 "sig_eps": [0, 0.012, 0.01, 0.01 ],
                 "scale"  : [6]
                 }

params_ranges_= {"mu_z"   : [0,   1, 0.2  ,  0.01 ],
                 "sig_z"  : [0,   1, 0.2  ,  0.01 ],
                 "bet_z"  : [0 , 25,  20  ,  2.   ],
                 "bet_xy" : [0,   2,   1. ,  1.   ],
                 "alpha"  : [0,   2,   1. ,  0.01 ],
                 "sig_eps": [0, 0.012, 0.01, 0.01 ],
                 "scale"  : [6]
                 }

param_scales = {"mu_z"   :  1,
                "sig_z"  :  1,
                "bet_z"  : 25,
                "bet_xy" :  2,
                "alpha"  :  2,}

paramscaler = ParamScaler(param_scales)

model_name           = 'JNet_239_x6_param_cross_attention'
hidden_channels_list = [16, 32, 64, 128, 256]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False
params               = {"mu_z"   : 0.2  , 
                        "sig_z"  : 0.2  , 
                        "bet_z"  :  20. , 
                        "bet_xy" :   1. , 
                        "alpha"  :   1. , 
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

optimizer            = optim.Adam(params, lr = 1e-4)
#scheduler            = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
scheduler            = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = warmup_func)
loss_fn              = nn.MSELoss()
midloss_fn           = nn.BCELoss()
param_loss_fn        = nn.MSELoss()

train_dataset = LabelandBlurParamsDataset(folderpath           = "_var_num_beadsdataset2"                 ,
                                          size                 = (1200, 500, 500)                         ,
                                          cropsize             = original_cropsize                        ,
                                          I                    = 200                                      ,
                                          low                  = 0                                        ,
                                          high                 = 16                                       ,
                                          imaging_function     = JNet.image                               ,
                                          imaging_params_range = params_ranges                            ,
                                          validation_params    = gen_imaging_parameters(params_ranges)    ,
                                          device               = device                                   ,
                                          is_train             = True                                     ,
                                          mask                 = True                                     ,
                                          mask_size            = [1, 10, 10]                              ,
                                          mask_num             = 30                                       ,
                                          surround             = surround                                 ,
                                          surround_size        = surround_size                            ,
                                          seed                 = 907                                      ,
                                          )
val_dataset   = LabelandBlurParamsDataset(folderpath           = "_var_num_beadsdataset2"                 ,
                                          size                 = (1200, 500, 500)                         ,
                                          cropsize             = original_cropsize                        ,
                                          I                    = 20                                       ,
                                          low                  = 16                                       ,
                                          high                 = 19                                       ,
                                          imaging_function     = JNet.image                               ,
                                          imaging_params_range = params_ranges                            ,
                                          validation_params    = gen_imaging_parameters(params_ranges_)   ,
                                          device               = device                                   ,
                                          is_train             = False                                    ,
                                          mask                 = False                                    ,
                                          surround             = surround                                 ,
                                          surround_size        = surround_size                            ,
                                          seed                 = 907                                      ,
                                          )

augment_param     = {"mask"          : True             , 
                     "mask_size"     : [1, 10, 10]      , 
                     "mask_num"      : 30               , 
                     "surround"      : surround         , 
                     "surround_size" : surround_size    ,
                     "original_size" : original_cropsize,
                     "cropsize"      : image_size[2:]   ,}
augment     = Augmentation(augment_param)
augment_param.update(mask=False)
val_augment = Augmentation(augment_param)

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

train_loop(
           n_epochs         = 100                  , ####
           optimizer        = optimizer            ,
           model            = JNet                 ,
           loss_fn          = loss_fn              ,
           param_loss_fn    = param_loss_fn        ,
           train_loader     = train_data           ,
           val_loader       = val_data             ,
           device           = device               ,
           path             = 'model'              ,
           savefig_path     = 'train'              ,
           model_name       = model_name           ,
           param_normalize  = paramscaler.normalize,
           augment          = augment              ,
           val_augment      = val_augment          ,
           partial          = partial              ,
           scheduler        = scheduler            ,
           es_patience      = 100                  ,
           reconstruct      = False                ,
           check_middle     = False                ,
           midloss_fn       = midloss_fn           ,
           is_randomblur    = True                 ,
           loss_weight      = 0                    ,
           qloss_weight     = 0                    ,
           paramloss_weight = 1                    ,
           )