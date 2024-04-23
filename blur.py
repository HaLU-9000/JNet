import torch
import model_new as model
import argparse
from train_loop import imagen_instantblur
from dataset import Vibrate
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import tifffile
import nd2
import json
import utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('org_folder' )
parser.add_argument('save_folder')
parser.add_argument('model_name' )
parser.add_argument('-image_name')
parser.add_argument('-keyword'   )
parser.add_argument('--vibrate', action="store_true")
args = parser.parse_args()
configs = open(os.path.join("experiments/configs",f"{args.model_name}.json"))
configs = json.load(configs)
params  = configs["params"]
# load model 
model_name = args.model_name
params     = configs["params"]
device = params["device"]

print(f"Training on device {device}.")
JNet = model.JNet(params)
JNet = JNet.to(device = device)
JNet.load_state_dict(torch.load(f'model/{configs["pretrained_model"]}.pt'),
                     strict=False)
JNet.eval()

if args.image_name is not None:
    items_new = [args.image_name]
else:
    items = os.listdir(args.org_folder)
    items_new = []
    for item in items:
        if args.keyword in item:
            items_new.append(item)
    items_new.sort()
os.makedirs(args.save_folder, exist_ok=True)

if args.vibrate:
    vibrate = Vibrate(vibration_params=configs["vibration"])
    vibrate.set_arbitrary_step(100)
else:
    vibrate = lambda x,y:x

for item in items_new:
    label = tifffile.imread(args.org_folder + "/" + item)
    s = label.shape[-1]//3
    label = torch.tensor(label[None, None, :, :, :s]/((2**16 - 1))).to(device)
    image = imagen_instantblur(JNet, label, None, None).detach().cpu()
    image = vibrate(image,True)[0, 0].numpy()
    image = (image * (2**16 - 1)).astype(np.uint16)
    tifffile.imwrite(args.save_folder+ "/"\
                     + utils.get_basename(item)\
                     + "_0.tif", image)
    del(image)