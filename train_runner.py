import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import timm.scheduler

import model_new as model
from dataset import RandomCutDataset
from dataset import ParamScaler, Augmentation
from train_loop import train_loop, ElasticWeightConsolidation

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

parser = argparse.ArgumentParser(description='Pretraining model.')
parser.add_argument('model_name')
args   = parser.parse_args()

configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"]
train_dataset_params = configs["pretrain_dataset"]
val_dataset_params   = configs["pretrain_val_dataset"]
train_loop_params    = configs["pretrain_loop"]
params["device"]     = device

JNet = model.JNet(params)
JNet = JNet.to(device = device)
train_params = JNet.parameters()

lr = train_loop_params['lr']

optimizer = optim.Adam(train_params, lr = lr)
scheduler = timm.scheduler.PlateauLRScheduler(
    optimizer      = optimizer   ,
    patience_t     = 10          ,
    warmup_lr_init = lr * 0.1    ,
    warmup_t       = 10          ,
    )
        
train_dataset = RandomCutDataset(
    folderpath    = train_dataset_params["folderpath"]   ,
    imagename     = train_dataset_params["imagename"]    , 
    labelname     = train_dataset_params["labelname"]    ,
    size          = train_dataset_params["size"]         ,
    cropsize      = train_dataset_params["cropsize"]     , 
    I             = train_dataset_params["I"]            ,
    low           = train_dataset_params["low"]          ,
    high          = train_dataset_params["high"]         ,
    scale         = train_dataset_params["scale"]        ,  ## scale
    mask          = train_dataset_params["mask"]         ,
    mask_size     = train_dataset_params["mask_size"]    ,
    mask_num      = train_dataset_params["mask_num"]     ,  #( 1% of image)
    surround      = train_dataset_params["surround"]     ,
    surround_size = train_dataset_params["surround_size"],
    )
val_dataset   = RandomCutDataset(
    folderpath    = val_dataset_params["folderpath"]   ,
    imagename     = val_dataset_params["imagename"]    , 
    labelname     = val_dataset_params["labelname"]    ,
    size          = val_dataset_params["size"]         ,
    cropsize      = val_dataset_params["cropsize"]     , 
    I             = val_dataset_params["I"]            ,
    low           = val_dataset_params["low"]          ,
    high          = val_dataset_params["high"]         ,
    scale         = val_dataset_params["scale"]        ,  ## scale
    mask          = val_dataset_params["mask"]         ,
    mask_size     = val_dataset_params["mask_size"]    ,
    mask_num      = val_dataset_params["mask_num"]     ,  #( 1% of image)
    surround      = val_dataset_params["surround"]     ,
    surround_size = val_dataset_params["surround_size"],
    seed          = val_dataset_params["seed"]         ,
    )       

train_data  = DataLoader(
    train_dataset                 ,
    batch_size  =train_loop_params["batch_size"],
    shuffle     = True            ,
    pin_memory  = True            ,
    num_workers = os.cpu_count()  ,
    )
val_data    = DataLoader(
    val_dataset                   ,
    batch_size  = train_loop_params["batch_size"],
    shuffle     = False           ,
    pin_memory  = True            ,
    num_workers = os.cpu_count()  ,
    )

print(f'============= model {configs["pretrained_model"]} train started =============')
model_path = 'model'
train_loop(
    n_epochs             = train_loop_params["n_epochs"      ]  ,
    loss_fn              = eval(train_loop_params["loss_fn"])   ,
    path                 = train_loop_params["path"]            ,
    savefig_path         = train_loop_params["savefig_path"]    ,
    partial              = eval(train_loop_params["partial"])   ,
    ewc                  = train_loop_params["ewc"]             ,
    es_patience          = train_loop_params["es_patience"   ]  ,
    reconstruct          = train_loop_params["reconstruct"   ]  ,
    is_instantblur       = train_loop_params["is_instantblur"]  ,
    is_vibrate           = train_loop_params["is_vibrate"    ]  ,
    loss_weight          = train_loop_params["loss_weight"   ]  ,
    qloss_weight         = train_loop_params["qloss_weight"  ]  ,
    model                = JNet                                 ,
    model_name           = configs["pretrained_model"]          ,
    params               = params                               ,
    train_dataset_params = train_dataset_params                 ,
    val_dataset_params   = val_dataset_params                   ,
    train_loader         = train_data                           ,
    val_loader           = val_data                             ,
    device               = device                               ,
    optimizer            = optimizer                            ,
    scheduler            = scheduler                            ,
    )