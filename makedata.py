import numpy as np
from numpy.random import randint, choice, randn
import torch
import skimage.morphology
from skimage.morphology import ball, octahedron, cube
import elasticdeform as deform

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

def make_data(num, datasize = (128, 128, 128)):
    data = np.zeros(datasize)
    r_l  = [randint(20, 21)                        for _ in range(num)]
    s_l  = [choice(['ball', 'octahedron', 'cube']) for _ in range(num)]
    z_l  = [randint(0, datasize[0])                for _ in range(num)]
    x_l  = [randint(0, datasize[1])                for _ in range(num)]
    y_l  = [randint(0, datasize[2])                for _ in range(num)]
    d_l  = [randn(3, 5, 5, 5)                      for _ in range(num)]
    for r, s, z, x, y, d in zip(r_l, s_l, z_l, x_l, y_l, d_l):
        form  = getattr(skimage.morphology, s)(r).astype(np.float32)
        form  = deform.deform_grid(X            = form ,
                                   displacement = d   ,)
        form  = form > 0.5
        z_max = min(z + form.shape[0], datasize[0])
        x_max = min(x + form.shape[1], datasize[1])
        y_max = min(y + form.shape[2], datasize[2])
        data[z : z + form.shape[0],
             x : x + form.shape[1],
             y : y + form.shape[2],]\
        += \
        form[0 : z_max - z        ,
             0 : x_max - x        ,
             0 : y_max - y        ,]
    data = data > 0
    data = torch.from_numpy(data)
    data = data.unsqueeze(0)
    return data

def make_beads_data(num, datasize = (128, 128, 128)):
    data = np.zeros(datasize)
    r_l  = [randint(15, 25)                        for _ in range(num)]
    s_l  = [choice(['ball', 'octahedron', 'cube']) for _ in range(num)]
    z_l  = [randint(0, datasize[0])                for _ in range(num)]
    x_l  = [randint(0, datasize[1])                for _ in range(num)]
    y_l  = [randint(0, datasize[2])                for _ in range(num)]
    d_l  = [randn(3, 5, 5, 5)                      for _ in range(num)]
    for r, s, z, x, y, d in zip(r_l, s_l, z_l, x_l, y_l, d_l):
        form  = getattr(skimage.morphology, s)(r).astype(np.float32)
        form  = deform.deform_grid(X=form, displacement=d*3,)
        form  = form > 0.5
        z_max = min(z + form.shape[0], datasize[0])
        x_max = min(x + form.shape[1], datasize[1])
        y_max = min(y + form.shape[2], datasize[2])
        data[z : z + form.shape[0],
             x : x + form.shape[1],
             y : y + form.shape[2],]\
        += \
        form[0 : z_max - z        ,
             0 : x_max - x        ,
             0 : y_max - y        ,]
    data = data > 0
    data = torch.from_numpy(data)
    data = data.unsqueeze(0)
    return data