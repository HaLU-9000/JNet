import numpy as np
import torch
from dataset import Blur
from utils import save
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
label = torch.load('/home/haruhiko/Documents/JNet/dataset/make_data_1e7.pt').float()
#label = label.to('cuda')
#print(label.shape)
#
for i in range(6):
    minlab = label[:, :, i * 256 : (i + 1) * 256, :].clone()
    num    = minlab.shape[1] * minlab.shape[2] * minlab.shape[3] // 128 ** 3
    save(minlab, 128, 128, 128, 'datasetpath', label=True, num=i*num)
for scale in (2, 4, 8):
    model = Blur(scale   = scale ,
                 z       = 141   ,
                 x       = 7     ,
                 y       = 7     ,
                 mu_z    = 0.2   ,
                 sig_z   = 0.2   , 
                 bet_xy  = 6.    ,
                 bet_z   = 35.   ,
                 sig_eps = 0.02  ,)
        #model.to('cuda')
    model.eval
    print(f'model load done (x{scale})')
    for i in range(6):
        minlab = label[:, :, i * 256 : (i + 1) * 256, :].clone()
        num    = minlab.shape[1] * minlab.shape[2] * minlab.shape[3] // 128 ** 3
        blur = model(minlab)
        print(f'blur done (x{scale})({5 - i} remains...)')
        save(blur, 128, 128, 128, 'datasetpath', name=f'_x{scale}', scale=scale, num=i*num)
    print(f'save done (x{scale})')