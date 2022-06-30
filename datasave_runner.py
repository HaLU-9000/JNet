import numpy as np
import torch
from dataloader import Blur
from utils import save
label = torch.load('/home/haruhiko/Documents/JNet/dataset/make_data_1e7.pt').float()
#save(label, 128, 128, 128, 'datasetpath', label=True)

for scale in (2, 4, 8, 12):
    model = None
    model = Blur(scale   = scale ,
                 z       = 141   ,
                 x       = 7     ,
                 y       = 7     ,
                 mu_z    = 0.2   ,
                 sig_z   = 0.2   ,
                 bet_xy  = 6.    ,
                 bet_z   = 35.   ,
                 sig_eps = 0.02  ,)
    model.eval
    print(f'model load done (x{scale})')
    blur = model(label)
    print(f'blur done (x{scale})')
    save(blur, 128, 128, 128, 'datasetpath', name=f'_x{scale}', scale=scale)
    print(f'save done (x{scale})')

