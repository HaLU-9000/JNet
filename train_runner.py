import os
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model as model
from dataset import LabelandBlurParamsDataset, gen_imaging_parameters
from train_loop import train_loop

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]

params_ranges = {"mu_z"   : [0,   1, 0.2  ,  0.2 ],
                 "sig_z"  : [0,   1, 0.2  ,  0.2 ],
                 "bet_z"  : [0 , 40,  20  ,  3   ],
                 "bet_xy" : [0,   2,   1. ,  0.25],
                 "alpha"  : [0, 100,  10  ,  5.  ],
                 "sig_eps": [0, 0.02, 0.01  , 0.0025],
                 "scale"  : [1, 2, 4, 8, 12      ]
                 }

model_name           = 'JNet_192_x6_randomblur'
hidden_channels_list = [16, 32, 64, 128, 256]
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
partial              = None #(56, 184)
superres = True if scale > 1 else False
params               = {"mu_z"   : 0.2    ,
                        "sig_z"  : 0.2    ,
                        "bet_z"  : 23.5329,
                        "bet_xy" : 1.00000,
                        "alpha"  : 0.9544 ,
                        "sig_eps": 0.01   ,
                        "scale"  : 6
                        }

JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  params                = params               ,
                  superres              = superres             ,
                  reconstruct           = False                ,
                  apply_vq              = False                ,
                  )
JNet = JNet.to(device = device)
#JNet.load_state_dict(torch.load('model/JNet_83_x1_partial.pt'), strict=False)
params = [i for i in JNet.parameters()][:-4]
#params = JNet.parameters()
optimizer            = optim.Adam(params, lr = 1e-4)
scheduler            = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
loss_fn              = nn.BCELoss()
midloss_fn           = nn.BCELoss()


train_dataset = LabelandBlurParamsDataset(folderpath           = "newrandomdataset"                       ,
                                          size                 = (1200, 500, 500)                         ,
                                          cropsize             = (240, 112, 112)                          ,
                                          I                    = 200                                       ,
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
val_dataset   = LabelandBlurParamsDataset(folderpath           = "newrandomdataset"                       ,
                                          size                 = (1200, 500, 500)                         ,
                                          cropsize             = (240, 112, 112)                          ,
                                          I                    = 10                                       ,
                                          low                  = 16                                       ,
                                          high                 = 19                                       ,
                                          imaging_function     = JNet.image                               ,
                                          imaging_params_range = params_ranges                            ,
                                          validation_params    = gen_imaging_parameters(params_ranges)    ,
                                          device               = device                                   ,
                                          is_train             = False                                    ,
                                          mask                 = False                                    ,
                                          surround             = surround                                 ,
                                          surround_size        = surround_size                            ,
                                          seed                 = 907                                      ,
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
           es_patience  = 15         ,
           tau_init     = 1          ,
           tau_lb       = 1          , 
           tau_sche     = 1          ,
           reconstruct  = False      ,
           check_middle = False      ,
           midloss_fn   = midloss_fn ,
           is_randomblur=True        ,
           )