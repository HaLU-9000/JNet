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
from dataset import RealDensityDataset, RandomCutDataset
from   train_loop import train_loop, ElasticWeightConsolidation
from inference import PretrainingInference

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

parser = argparse.ArgumentParser(description='Pretraining model.')
parser.add_argument('model_name')
args   = parser.parse_args()

configs = open(os.path.join("experiments/configs",f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"]
train_dataset_params = configs["train_dataset"]
ewc_dataset_params   = configs["pretrain_dataset"]
val_dataset_params   = configs["val_dataset"]
train_loop_params    = configs["train_loop"]

#infer = PretrainingInference(args.model_name, pretrain=True)
#results = infer.get_result(10)
#threshold = infer.threshold_argmax_f1score(results)
#params["threshold"] = threshold

#with open(os.path.join("experiments/configs",
#                       f"{args.model_name}.json"), "w") as f:
#    json.dump(configs, f, indent=4)

params["reconstruct"]     = True
params["apply_vq"]        = True
#params["use_x_quantized"] = True

train_dataset = RealDensityDataset(
    folderpath      = train_dataset_params["folderpath"     ]        ,
    scorefolderpath = train_dataset_params["scorefolderpath"]        ,
    imagename       = train_dataset_params["imagename"      ]        ,
    size            = train_dataset_params["size"           ]        , # size after segmentation
    cropsize        = train_dataset_params["cropsize"       ]        , # size after segmentation
    I               = train_dataset_params["I"              ]        ,
    low             = train_dataset_params["low"            ]        ,
    high            = train_dataset_params["high"           ]        ,
    scale           = train_dataset_params["scale"          ]        , ## scale
    train           = train_dataset_params["train"          ]        ,
    mask            = train_dataset_params["mask"           ]        ,
    mask_num        = train_dataset_params["mask_num"       ]        ,
    mask_size       = train_dataset_params["mask_size"      ]        ,
    surround        = train_dataset_params["surround"       ]        ,
    surround_size   = train_dataset_params["surround_size"  ]        ,
    score           = torch.load(train_dataset_params["score_path"]) ,
    )

val_dataset   = RealDensityDataset(
    folderpath      = val_dataset_params["folderpath"     ]         ,
    scorefolderpath = val_dataset_params["scorefolderpath"]         ,
    imagename       = val_dataset_params["imagename"      ]         ,
    size            = val_dataset_params["size"           ]         , # size after segmentation
    cropsize        = val_dataset_params["cropsize"       ]         ,
    I               = val_dataset_params["I"              ]         ,
    low             = val_dataset_params["low"            ]         ,
    high            = val_dataset_params["high"           ]         ,
    scale           = val_dataset_params["scale"          ]         ,
    train           = val_dataset_params["train"          ]         ,
    mask            = val_dataset_params["mask"           ]         ,
    mask_size       = val_dataset_params["mask_size"      ]         ,
    mask_num        = val_dataset_params["mask_num"       ]         ,
    surround        = val_dataset_params["surround"       ]         ,
    surround_size   = val_dataset_params["surround_size"  ]         ,
    seed            = val_dataset_params["seed"           ]         ,
    score           = torch.load(val_dataset_params["score_path"])  ,
    )

train_data  = DataLoader(
    train_dataset                                 ,
    batch_size  = train_loop_params["batch_size"] ,
    shuffle     = True                            ,
    pin_memory  = True                            ,
    num_workers = os.cpu_count()                  ,
    )

val_data    = DataLoader(
    val_dataset                                   ,
    batch_size  = train_loop_params["batch_size"] ,
    shuffle     = False                           ,
    pin_memory  = True                            ,
    num_workers = os.cpu_count()                  ,
    )

JNet = model.JNet(params)
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{configs["pretrained_model"]}.pt'),
                     strict=False)
#print([i for i in JNet.parameters()][-4:])

train_params = JNet.parameters()
for param in JNet.image.blur.parameters():
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
    imagename     = ewc_dataset_params["imagename"]    , 
    labelname     = ewc_dataset_params["labelname"]    ,
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
print(f"============= model {args.model_name} train started =============")
train_loop(
    n_epochs         = train_loop_params["n_epochs"      ]  , ####
    loss_fn          = eval(train_loop_params["loss_fn"])   ,
    path             = train_loop_params["path"]            ,
    savefig_path     = train_loop_params["savefig_path"]    ,
    ewc              = eval(train_loop_params["partial"])   ,
    partial          = train_loop_params["ewc"]             ,
    es_patience      = train_loop_params["es_patience"   ]  ,
    reconstruct      = train_loop_params["reconstruct"   ]  ,
    is_instantblur   = train_loop_params["is_instantblur"]  ,
    is_vibrate       = train_loop_params["is_vibrate"    ]  ,
    loss_weight      = train_loop_params["loss_weight"   ]  ,
    qloss_weight     = train_loop_params["qloss_weight"  ]  ,
    ploss_weight     = train_loop_params["ploss_weight"  ]  ,
    model            = JNet                                 ,
    model_name       = args.model_name                      ,
    params           = params                               ,
    train_dataset_params = train_dataset_params             ,
    val_dataset_params   = val_dataset_params               ,
    train_loader     = train_data                           ,
    val_loader       = val_data                             ,
    device           = device                               ,
    optimizer        = optimizer                            ,
    scheduler        = scheduler                            ,
    )
