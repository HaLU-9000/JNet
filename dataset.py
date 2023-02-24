from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import lognorm

from utils import mask_, surround_mask_
from model import ImagingProcess


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


class Crop:
    def __init__(self, coord:list, cropsize:list):
        self.coord = coord
        self.csize = cropsize
    def __call__(self,x:torch.Tensor):
       return x[0 : , self.coord[0] : self.coord[0] + self.csize[0],
                      self.coord[1] : self.coord[1] + self.csize[1],
                      self.coord[2] : self.coord[2] + self.csize[2]].detach().clone()


class RandomCutDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (image and label)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly cropped/rotated image and label
    folderpath : large data path ("randomdata" in this repo)
    imagename : "0001***.pt" `s "**" part. (e.g. "_x1")
    labelname : "0001***.pt" `s "**" part. (e.g. "_label")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    scale: scale (should be same as [imagename]'s int part.)
    '''
    def __init__(self, folderpath:str, imagename:str, labelname:str, 
                 size:list, cropsize:list, I:int, low:int, high:int, scale:int,
                 train=True, pretrain=True, mask=True,
                 mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[64, 8, 8],
                 seed=904):
        self.I             = I
        self.low           = low
        self.high          = high
        self.scale         = scale
        self.size          = size
        self.labels        = list(sorted(Path(folderpath).glob(f'*{labelname}.pt')))
        self.images        = list(sorted(Path(folderpath).glob(f'*{imagename}.pt')))
        self.csize         = cropsize
        self.ssize         = [cropsize[0]//scale, cropsize[1], cropsize[2]]
        self.train         = train
        self.mask          = mask
        self.mask_size     = mask_size
        self.mask_num      = mask_num
        self.surround      = surround
        self.surround_size = surround_size

        if train == False:
            np.random.seed(seed)
            self.indiceslist = self.gen_indices(I, low, high)
            self.coordslist  = self.gen_coords(I, size, cropsize, scale)

    def gen_indices(self, I, low, high):
        return np.random.randint(low, high, (I,))
    
    def gen_coords(self, I, size, cropsize, scale):
        zcoord = np.random.randint(0, size[0]-cropsize[0], (I,))
        xcoord = np.random.randint(0, size[1]-cropsize[1], (I,))
        ycoord = np.random.randint(0, size[2]-cropsize[2], (I,))
        return np.array([zcoord, xcoord, ycoord]), np.array([zcoord // scale, xcoord, ycoord])

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def __getitem__(self, idx):
        if self.train:
            idx              = self.gen_indices(1, self.low, self.high).item()#            print('idx ', idx)
            lcoords, icoords = self.gen_coords(1, self.size, self.csize, self.scale)
            lcoords, icoords = lcoords[:, 0], icoords[:, 0]
            image, i, j      = Rotate(    )(Crop(icoords, self.ssize
                                                )(torch.load(self.images[idx]))) # .unsqueeze(0) for beadslikedata5
            label, _, _      = Rotate(i, j)(Crop(lcoords, self.csize
                                                )(torch.load(self.labels[idx])))
            
            image = self.apply_mask(self.mask, image, self.mask_size, self.mask_num)
            image = self.apply_surround_mask(self.surround, image, self.surround_size)
        else:
            _idx    = self.indiceslist[idx]  # convert idx to [low] ~[high] number
            icoords = self.coordslist[1][:, idx]
            lcoords = self.coordslist[0][:, idx]
            image   = Crop(icoords, self.ssize)(torch.load(self.images[_idx])) # .unsqueeze(0) for beadslikedata5
            label   = Crop(lcoords, self.csize)(torch.load(self.labels[_idx]))
            image = self.apply_surround_mask(self.surround, image, self.surround_size)
        return image, label

    def __len__(self):
        return self.I
    

class RandomBlurDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (label)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly blurred/cropped/rotated image and label
    folderpath : large data path ("randomdata" in this repo)
    imagename : "0001***.pt" `s "**" part. (e.g. "_x1")
    labelname : "0001***.pt" `s "**" part. (e.g. "_label")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    scale: scale 
    '''
    def __init__(self, folderpath:str, labelname:str, 
                 size:list, cropsize:list, filtersize:list, 
                 I:int, low:int, high:int, scale:int,
                 sig_mu_z, sig_sig_z, mu_bet_xy, sig_bet_xy,
                 mu_bet_z, sig_bet_z, mu_alpha,  sig_alpha,
                 device, train=True, mask=True, varid_params=[],
                 mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[64, 8, 8],
                 seed=904,):
        self.I             = I
        self.low           = low
        self.high          = high
        self.scale         = scale
        self.size          = size
        self.labels        = list(sorted(Path(folderpath).glob(f'*{labelname}.pt')))
        self.csize         = cropsize
        self.ssize         = [cropsize[0]//scale, cropsize[1], cropsize[2]]
        self.train         = train
        self.mask          = mask
        self.mask_size     = mask_size
        self.mask_num      = mask_num
        self.surround      = surround
        self.surround_size = surround_size
        self.varid_params  = varid_params
        self.sig_mu_z      = sig_mu_z  
        self.sig_sig_z     = sig_sig_z 
        self.mu_bet_xy     = mu_bet_xy 
        self.sig_bet_xy    = sig_bet_xy
        self.mu_bet_z      = mu_bet_z  
        self.sig_bet_z     = sig_bet_z 
        self.mu_alpha      = mu_alpha  
        self.sig_alpha     = sig_alpha 
        z, x, y            = filtersize
        self.imaging       = ImagingProcess(scale, z, x, y, *varid_params,
                                            device)
        if train == False:
            np.random.seed(seed)
            self.indiceslist = self.gen_indices(I, low, high)
            self.coordslist  = self.gen_coords(I, size, cropsize, scale)

    def gen_indices(self, I, low, high):
        return np.random.randint(low, high, (I,))
    
    def gen_coords(self, I, size, cropsize,):
        zcoord = np.random.randint(0, size[0]-cropsize[0], (I,))
        xcoord = np.random.randint(0, size[1]-cropsize[1], (I,))
        ycoord = np.random.randint(0, size[2]-cropsize[2], (I,))
        return np.array([zcoord, xcoord, ycoord])

    def gen_params(self):
        mu_z   = abs(np.random.normal(0, self.sig_mu_z))
        sig_z  = abs(np.random.normal(0, self.sig_sig_z))
        bet_xy = abs(np.random.normal(self.mu_bet_xy, self.sig_bet_xy))
        bet_z  = abs(np.random.normal(self.mu_bet_z , self.sig_bet_z ))
        alpha  = abs(np.random.normal(self.mu_alpha , self.sig_alpha))
        return [mu_z, sig_z, bet_xy, bet_z, alpha]

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def __getitem__(self, idx):
        if self.train:
            idx     = self.gen_indices(1, self.low, self.high).item()#            print('idx ', idx)
            lcoords = self.gen_coords(1, self.size, self.csize,)
            lcoords = lcoords[:, 0]
            label, _, _  = Rotate(    )(Crop(lcoords, self.csize
                                            )(torch.load(self.labels[idx])))
            params = self.gen_params()
            image  = self.imaging.sample(label, *params)
            image  = self.apply_mask(self.mask, image, self.mask_size,
                                     self.mask_num)
            image  = self.apply_surround_mask(self.surround, image,
                                              self.surround_size)

        else:
            _idx    = self.indiceslist[idx]  # convert idx to [low] ~[high] number
            lcoords = self.coordslist[0][:, idx]
            label   = Crop(lcoords, self.csize)(torch.load(self.labels[_idx]))
            image   = self.imaging(label, *self.varid_params)
            image   = self.apply_surround_mask(self.surround, image,
                                               self.surround_size)

        return image, label, params

    def __len__(self):
        return self.I


class RealDensityDataset(Dataset):
    '''
    input  : 4d torch.tensor (large (like 768**3) size) (image)
    output : 4d torch.tensor (small (like 128**3) size)
             ([channels, z_size, x_size, y_size]) 
             of randomly cropped/rotated image
    folderpath : large data path ("randomdata" in this repo)
    imagename : "0001***.pt" `s "**" part. (e.g. "_x1")
    I : sample size. Returns I samples. (e.g. 200)
    low, high : use [low]th ~ [high]th files in folderpath as data.
    scale: scale (should be same as [imagename]'s int part.)

    algorithm
    (init)
    1. calculate score by conv filter
    2. normalize score to [0, 1]
    (__getitem__)
    3. r ~ uniform(0,1)
    4. accept | if r < score
       reject | else
    '''
    def __init__(self, folderpath:str, scorefolderpath:str, imagename:str,
                 size:list, cropsize:list, I:int, low:int, high:int, scale:int,
                 train=True, mask=True, score=None, score_saving=True,
                 mask_size=[10, 10, 10], mask_num=1,
                 surround=True, surround_size=[64, 8, 8],
                 seed=904):
        self.I             = I
        self.low           = low
        self.high          = high
        self.scale         = scale
        self.size          = size
        self.ssize         = [size[0]//scale, size[1], size[2]]
        self.imagename     = imagename
        self.images        = list(sorted(Path(folderpath).glob(f'*{imagename}.pt')))
        self.csize         = cropsize
        self.scsize        = [cropsize[0]//scale, cropsize[1], cropsize[2]]
        self.train         = train
        self.mask          = mask
        self.mask_num      = mask_num
        self.mask_size     = mask_size
        self.surround      = surround
        self.surround_size = surround_size
        self.icoords_size  = [self.ssize[0] - self.scsize[0] + 1,
                              self.ssize[1] - self.scsize[1] + 1,
                              self.ssize[2] - self.scsize[2] + 1,]
        
        if score is None:
            self.scores = self.gen_scores(self.images, self.icoords_size, self.scsize)
            if score_saving:
                self.save_scores(self.scores, scorefolderpath)
        else:
            self.scores = score
        if train == False:
            np.random.seed(seed)
            self.indiceslist = self.gen_indices(I, low, high)
            self.coordslist  = self.gen_coords(I, self.icoords_size)

    def gen_scores(self, images, icoords_size, scsize):
        _scores = torch.zeros((len(images), 1, *icoords_size))
        for n, i in enumerate(images):
            print(f'(init) calcurating the score...({n+1}/{len(images)})')
            _score = F.conv3d(input   = torch.load(i)          ,
                              weight  = torch.ones(1,1,*scsize),
                              stride  = 1                      ,
                              padding = 0                      ,)
            _scores[n] = _score
        _fscores = _scores.flatten()
        _scores = (_scores - torch.min(_fscores))            \
                / ((torch.max(_fscores) - torch.min(_fscores)))
        return _scores

    def save_scores(self, scores, scorefolderpath):
        torch.save(scores, scorefolderpath+f'/{self.imagename}_score.pt')

    def gen_indices(self, I, low, high):
        return np.random.randint(low, high, (I,))
    
    def gen_coords(self, I, icoords_size):
        zcoord = np.random.randint(0, icoords_size[0], (I,))
        xcoord = np.random.randint(0, icoords_size[1], (I,))
        ycoord = np.random.randint(0, icoords_size[2], (I,))
        return np.array([zcoord, xcoord, ycoord])

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image

    def __getitem__(self, idx):
        if self.train:
            r, s = 1, 0
            while not r < s:
                idx     = self.gen_indices(1, self.low, self.high).item()
                icoords = self.gen_coords( 1 , self.icoords_size)
                icoords = icoords[:, 0]
                z, x, y = icoords
                r = np.random.uniform(0, 1)
                s = self.scores[idx, 0, z, x, y]
            image, _, _      = Rotate(    )(Crop(icoords, self.scsize
                                                )(torch.load(self.images[idx])))
            image = self.apply_mask(self.mask, image, self.mask_size, self.mask_num)
            image = self.apply_surround_mask(self.surround, image, self.surround_size)
        else:
            _idx    = self.indiceslist[idx]  # convert idx to [low] ~[high] number
            icoords = self.coordslist[:, idx]
            image   = Crop(icoords, self.scsize)(torch.load(self.images[_idx]))
            image = self.apply_surround_mask(self.surround, image, self.surround_size)
        return image, 0

    def __len__(self):
        return self.I

class DatasetUtils():
    def gen_indices(self, I, low, high):
        return np.random.randint(low, high, (I,))
    
    def gen_coords(self, I, size, cropsize, scale, label):
        zcoord = np.random.randint(0, size[0]-cropsize[0], (I,))
        xcoord = np.random.randint(0, size[1]-cropsize[1], (I,))
        ycoord = np.random.randint(0, size[2]-cropsize[2], (I,))
        if label is not None:
            return np.array([zcoord, xcoord, ycoord]), np.array([zcoord // scale, xcoord, ycoord])
        else:
            return np.array([zcoord // scale, xcoord, ycoord])

    def apply_mask(self, mask, image, mask_size, mask_num):
        if mask:
            image = mask_(image, mask_size, mask_num)
        return image

    def apply_surround_mask(self, surround, image, surround_size):
        if surround:
            image = surround_mask_(image, surround_size)
        return image