from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from torch.utils.data import Dataset


class Blur(nn.Module):
    def __init__(self, scale, z, x, y, mu_z, sig_z, bet_xy, bet_z, sig_eps,):
        super().__init__()
        self.scale   = scale
        self.z       = z
        self.x       = x
        self.y       = y
        self.mu_z    = mu_z
        self.sig_z   = sig_z
        self.bet_xy  = bet_xy
        self.bet_z   = bet_z
        self.sig_eps = sig_eps
        self.zd, self.xd, self.yd   = self.distance(z, x, y)
        self.alf     = self.gen_alf(self.zd, self.xd, self.yd, bet_xy, bet_z)
        
    def distance(self, z, x, y):
        [zd, xd, yd] = [torch.zeros(1, 1, z, x, y,) for _ in range(3)]
        for k in range(-z // 2, z // 2 + 1):
            zd[:, :, k + z // 2, :, :,] = k ** 2
        for i in range(-x // 2, x // 2 + 1):
            xd[:, :, :, i + x // 2, :,] = i ** 2
        for j in range(-y // 2, y // 2 + 1):
            yd[:, :, :, :, j + y // 2,] = j ** 2
        return zd, xd, yd

    def gen_alf(self, zd, xd, yd, bet_z, bet_xy):
        d_2 = self.zd / self.bet_z ** 2 + (self.xd + self.yd) / self.bet_xy ** 2
        return torch.exp(-d_2 / 2) / (torch.pi * 2) ** 0.5

    def forward(self, inp):
        inp  = inp.unsqueeze(0)
        pz0  = dist.LogNormal(loc   = self.mu_z  * torch.ones_like(inp),
                              scale = self.sig_z * torch.ones_like(inp),)       
        rec  = inp * pz0.sample()
        rec  = F.conv3d(input   = rec                            ,
                        weight  = self.alf                       ,
                        stride  = (self.scale, 1, 1)             ,
                        padding = ((self.z - self.scale + 1) // 2, 
                                   (self.x) // 2                 , 
                                   (self.y) // 2                 ,))
        rec  = (rec - rec.min()) / (rec.max() - rec.min())
        prec = dist.Normal(loc   = rec         ,
                           scale = self.sig_eps,)
        rec  = prec.sample()
        rec  = rec.squeeze(0)
        return rec

class CustomDataset(Dataset):
    def __init__(self, ori_data, scale, zsize, xsize, ysize, z, x=1, y=1, bet_xy=1, bet_z=50,):
        model = Blur(scale   = scale ,
                     z       = z     ,
                     x       = x     ,
                     y       = y     ,
                     mu_z    = 0.2   ,
                     sig_z   = 0.2   ,
                     bet_xy  = bet_xy,
                     bet_z   = bet_z ,
                     sig_eps = 0.02   ,)
        blur_zsize  = zsize // scale
        true        = ori_data
        self.trues  = self.cut(true                  , zsize     , xsize, ysize).float()
        self.blurs  = self.cut(self.blur(true, model), blur_zsize, xsize, ysize)
    def blur(self, inp, model):
        blur = model(inp)
        blur = (blur - blur.min()) / (blur.max() - blur.min())
        return blur
    
    def cut(self, inp, zsize, xsize, ysize):
        """
        input  : 4d tensor ([channels, z_shape, x_shape, y_shape])
        output : 5d tensor ([patches, channels, z_size, x_size, y_size]) 
        """
        inp = inp.unfold(1, zsize , zsize ,)\
                 .unfold(2, xsize , xsize ,)\
                 .unfold(3, ysize , ysize ,)
        return inp.contiguous().view(-1, 1, zsize, xsize, ysize)

    def __getitem__(self, idx):
        return self.blurs[idx], self.trues[idx]
    
    def __len__(self):
        return self.blurs.shape[0]

class Rotate:
    def __init__(self, i=None, j=None):
        if i is not None:
            self.i = i
        else:
            self.i = random.choice([0,2])
        if j is not None:
            self.j = j
        else:
            self.j = random.choice([0,1,2,3])
    def __call__(self, x):
        return torch.rot90(torch.rot90(x, self.i, [1, 2]), self.j, [2, 3]), self.i, self.j

class PathDataset(Dataset):
    def __init__(self, folderpath, imagename, labelname='_label'):
        self.labels = list(sorted(Path(folderpath).glob(f'*{labelname}.npy')))
        self.images = list(sorted(Path(folderpath).glob(f'*{imagename}.npy')))
    def __getitem__(self, idx):
        image = torch.from_numpy(np.load(self.images[idx]))
        label = torch.from_numpy(np.load(self.labels[idx]))
        return image, label
    def __len__(self):
        return len(self.labels)

class RotateDataset(Dataset):
    def __init__(self, folderpath, imagename, labelname='_label'):
        self.labels = list(sorted(Path(folderpath).glob(f'*{labelname}.npy')))
        self.images = list(sorted(Path(folderpath).glob(f'*{imagename}.npy')))
    def __getitem__(self, idx):
        image, i, j = Rotate(    )(torch.from_numpy(np.load(self.images[idx])))
        label, _, _ = Rotate(i, j)(torch.from_numpy(np.load(self.labels[idx])))
        return image, label
    def __len__(self):
        return len(self.labels)

"""
class RandomCutDataset(Dataset):
    '''
    input  : 4d numpy.ndarray(label) ,   4d numpy array(blurred image) 
    output : 5d torch.tensor 
             ([patches, channels, z_size, x_size, y_size]) 
             of randomly cropped image and label
    '''
    def __init__(self, labels, images):
        
    def randomcut(self, image, zsize, xsize, ysize, i, j, k, scale=None):
        
        image[]
    def __getitem__(self, idx):
        label, i, j, k  = self.randomcut(label, zsize, xsize, ysize, scale=None)
        image, _, _, _  = self.randomcut(image, zsize, xsize, ysize, i, j, k)
        return image, label
    def __len__(self):
        return self.blurs.shape[0]
"""