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