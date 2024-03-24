import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import timm.scheduler

import model_new as model
from dataset import DensityDataset, RandomCutDataset

from   train_loop                           \
    import finetuning_loop,                 \
           ElasticWeightConsolidation,      \
           finetuning_with_align_model_loop

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

parser = argparse.ArgumentParser(description='Pretraining model.')
parser.add_argument('model_name')
parser.add_argument('--train_with_align', action="store_true")
parser.add_argument('-t', '--train_mode', default='old', choices=['all', 'encoder', 'decoder', 'old'])
args   = parser.parse_args()

configs = open(os.path.join("experiments/configs",f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"           ]
train_dataset_params = configs["train_dataset"    ]
ewc_dataset_params   = configs["pretrain_dataset" ]
val_dataset_params   = configs["val_dataset"      ]
train_loop_params    = configs["train_loop"       ]
vibration_params     = configs["vibration"        ]

#infer = PretrainingInference(args.model_name, pretrain=True)
#results = infer.get_result(10)
#threshold = infer.threshold_argmax_f1score(results)
#params["threshold"] = threshold

#with open(os.path.join("experiments/configs",
#                       f"{args.model_name}.json"), "w") as f:
#    json.dump(configs, f, indent=4)

train_dataset = DensityDataset(
    folderpath      = train_dataset_params["folderpath"   ] ,
    size            = train_dataset_params["size"         ] , # size after segmentation
    cropsize        = train_dataset_params["cropsize"     ] , # size after segmentation
    I               = train_dataset_params["I"            ] ,
    scale           = train_dataset_params["scale"        ] , ## scale
    train           = train_dataset_params["train"        ] ,
    mask            = train_dataset_params["mask"         ] ,
    mask_num        = train_dataset_params["mask_num"     ] ,
    mask_size       = train_dataset_params["mask_size"    ] ,
    surround        = train_dataset_params["surround"     ] ,
    surround_size   = train_dataset_params["surround_size"] ,
    )

val_dataset   = DensityDataset(
    folderpath      = val_dataset_params["folderpath"     ] ,
    size            = val_dataset_params["size"           ] , # size after segmentation
    cropsize        = val_dataset_params["cropsize"       ] ,
    I               = val_dataset_params["I"              ] ,
    scale           = val_dataset_params["scale"          ] ,
    train           = val_dataset_params["train"          ] ,
    mask            = val_dataset_params["mask"           ] ,
    mask_size       = val_dataset_params["mask_size"      ] ,
    mask_num        = val_dataset_params["mask_num"       ] ,
    surround        = val_dataset_params["surround"       ] ,
    surround_size   = val_dataset_params["surround_size"  ] ,
    seed            = val_dataset_params["seed"           ] ,
    )

train_data  = DataLoader(
    train_dataset                                 ,
    batch_size  = train_loop_params["batch_size"] ,
    shuffle     = True                            ,
    pin_memory  = True                            ,
    num_workers = 0#os.cpu_count()                ,
    )

val_data    = DataLoader(
    val_dataset                                   ,
    batch_size  = train_loop_params["batch_size"] ,
    shuffle     = False                           ,
    pin_memory  = True                            ,
    num_workers = 0#os.cpu_count()                ,
    )

params["reconstruct"]     = True
params["apply_vq"]        = True
params["use_x_quantized"] = True

JNet = model.JNet(params)
JNet = JNet.to(device = device)

if args.train_with_align:
    align_params   = configs["align_params"]
    deep_align_net = model.DeepAlignNet(align_params)
    deep_align_net = deep_align_net.to(device=device)

if args.train_mode == 'decoder' or args.train_mode == 'old':
    JNet.load_state_dict(torch.load(f'model/{configs["pretrained_model"]}.pt'),
                         strict=False)
if args.train_mode == 'all':
    JNet.load_state_dict(torch.load(f'model/{args.model_name}.pt'),
                        strict=False)

train_params = JNet.parameters()

if args.train_mode == 'decoder':
    for param in JNet.parameters():
        param.requires_grad = False
    for param in JNet.image.parameters():
        param.requires_grad = True
    for param in JNet.image.blur.parameters():
        param.requires_grad = False

if args.train_mode == 'encoder':
    for param in JNet.image.parameters():
        param.requires_grad = False

lr = train_loop_params["lr"]

optimizer            = optim.Adam(filter(lambda p: p.requires_grad, JNet.parameters()), lr = lr)
scheduler            = timm.scheduler.PlateauLRScheduler(
    optimizer      = optimizer   ,
    patience_t     = 10          ,
    warmup_lr_init = lr * 0.1    ,
    warmup_t       = 10          ,)

ewc_dataset   = RandomCutDataset(
    folderpath    = ewc_dataset_params["folderpath"]   ,
    size          = ewc_dataset_params["size"]         ,
    cropsize      = ewc_dataset_params["cropsize"]     , 
    I             = ewc_dataset_params["I"]            ,
    low           = ewc_dataset_params["low"]          ,
    high          = ewc_dataset_params["high"]         ,
    scale         = ewc_dataset_params["scale"]        ,  ## scale
    mask          = ewc_dataset_params["mask"]         ,
    mask_size     = ewc_dataset_params["mask_size"]    ,
    mask_num      = ewc_dataset_params["mask_num"]     ,  #( 1% of image)
    surround      = ewc_dataset_params["surround"]     ,
    surround_size = ewc_dataset_params["surround_size"],
    )

ewc_data    = DataLoader(
    ewc_dataset                   ,
    batch_size  = 1               ,
    shuffle     = True            ,
    pin_memory  = True            ,
    num_workers = 0#os.cpu_count()  ,
    )

if  train_loop_params["ewc"] != None:
    ewc = ElasticWeightConsolidation(
        model              = JNet                                       ,
        params             = params                                     ,
        vibration_params   = vibration_params                           ,
        prev_dataloader    = ewc_data                                   ,
        loss_fnx           = eval(configs["pretrain_loop"]["loss_fnx"]) ,
        loss_fnz           = eval(configs["pretrain_loop"]["loss_fnz"]) ,
        wx                 = configs["pretrain_loop"]["weight_x"]       ,
        wz                 = configs["pretrain_loop"]["weight_z"]       ,
        ewc_dataset_params = ewc_dataset_params                         ,
        init_num_batch     = 100                                        ,
        is_vibrate         = True                                       ,
        device             = device
        )
else:
    ewc = None

if args.train_with_align:
    print(f"============= model {args.model_name} train started " + \
          f"with {align_params['name']} =============")
    finetuning_with_align_model_loop(
        optimizer            = optimizer            ,
        model                = JNet                 ,
        align_model          = deep_align_net       ,
        train_loader         = train_data           ,
        val_loader           = val_data             ,
        device               = device               ,
        model_name           = args.model_name      ,
        ewc                  = ewc                  ,
        train_dataset_params = train_dataset_params ,
        train_loop_params    = train_loop_params    ,
        scheduler            = scheduler
    )

else:
    print(f"============= model {args.model_name} train started =============")
    finetuning_loop( ####
        optimizer            = optimizer            ,
        model                = JNet                 ,
        train_loader         = train_data           ,
        val_loader           = val_data             ,
        device               = device               ,
        model_name           = args.model_name      ,
        ewc                  = ewc                  ,
        train_dataset_params = train_dataset_params ,
        train_loop_params    = train_loop_params    ,
        vibration_params     = vibration_params     ,
        scheduler            = scheduler
    )