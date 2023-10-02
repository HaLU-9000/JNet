import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import RealDensityDataset, sequentialflip

import model_new as model

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Inference on device {device}.")

parser = argparse.ArgumentParser(description='inference with bead images.')
parser.add_argument('model_name')
args   = parser.parse_args()

configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs              = json.load(configs)
params               = configs["params"]

JNet = model.JNet(params)
JNet = JNet.to(device = device)
j   = 12
j_s = j // params["scale"]
i = 64

JNet.load_state_dict(torch.load(f'model/{args.model_name}.pt'), strict=False)
JNet.eval()
dirpath = "_beads_roi_extracted_stackreg"
images = [os.path.join(dirpath, f) for f in sorted(os.listdir(dirpath))]
print(images)
outputs = []
loss_fn = nn.MSELoss()
for image_name in images[:-1]:
    image_ = torch.load(image_name, map_location="cuda").to(torch.float32)
    image = image_#(torch.clip(image_, min=0.1, max=1.) - 0.1) / (1.0 - 0.1)
    outdict = JNet(image.to("cuda").unsqueeze(0))
    output  = outdict["enhanced_image"]
    output  = output.detach().cpu()
    reconst = outdict["reconstruction"]
    #loss    = loss_fn(reconst, image).item()
    qloss   = outdict["quantized_loss"]
    print("output ", torch.sum(output).item() * (0.05 * 0.05 * 0.05))
    outputs.append(torch.sum(output).item()* (0.05 * 0.05 * 0.05))
    reconst = reconst.squeeze(0).detach().cpu().numpy()
    #losses.append(loss)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.set_title('original image')
    ax2.set_title(f'reconstruct image\n{args.model_name}')
    plt.subplots_adjust(hspace=-0.0)
    ax1.imshow(image_[0, :, i, :].to(device='cpu'),
            cmap='gray', vmin=0.0, vmax=1.0, aspect=params["scale"])
    ax2.imshow(output[0, 0, :, i, :],
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    plt.savefig(f'result/{args.model_name}_new_{image_name[30:-3]}.png', format='png', dpi=250)
print(args.model_name)
print(np.mean(np.array(outputs)))