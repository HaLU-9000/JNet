import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from metrics import jiffs

class EarlyStopping():
    """
    path[str]: path you want to save your model
    name[str]: model name
    patience[int]: default = 10 
    modified from https://qiita.com/ku_a_i/items/ba33c9ce3449da23b503
    """
    def __init__(self, path, name, patience=10, verbose=False):
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.Inf
        self.path         = path
        self.name         = name
    def __call__(self, val_loss, model, condition=False):
        score = -val_loss
        if self.best_score is None: # 1st time
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score and condition: # earlystop (only if condition is True)
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter:{self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print('EarlyStopping!')
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0
    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'loss ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving models...')
        torch.save(model.state_dict(), f'{self.path}/{self.name}.pt')
        self.val_loss_min = val_loss

class ModelSizeEstimator():
    def __init__(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        self.size_all_mb = (param_size + buffer_size) / 1024**2
    def __call__(self):
        print('model size: {:.3f} MB'.format(self.size_all_mb))

def save(data, zsize, xsize, ysize, path, name='', label=True, scale=1, num=0):
    """
    input  : 4d pt.tensor
    output : 4d numpy.ndarray
    * not working well in x12. modify when needed.
    """
    if name:
        label = False
    if label:
        l = '_label'
    else:
        l = name    
    zsize = zsize // scale
    z_r   = data.shape[1] // zsize
    x_r   = data.shape[2] // xsize
    y_r   = data.shape[3] // ysize
    for k in range(z_r):
        for i in range(x_r):
            for j in range(y_r):
                pt = data[0 : , k * zsize : (k + 1) * zsize ,
                                i * xsize : (i + 1) * xsize ,
                                j * ysize : (j + 1) * ysize ,].clone()
                np.save(f'{path}/{str(num).zfill(4)}{l}', pt.detach().cpu().numpy())
                num += 1
    return num

def path_blur():
    return 0

def save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, device, I=0):
    flist = list(sorted(Path(folderpath).glob(f'*{labelname}.npy')))
    model = model.to(device)
    for i, label in enumerate(flist[I:]):
        label = torch.from_numpy(np.load(label))
#        if not Path(f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt').is_file():
        torch.save(label.float(),  f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt')
        blur = model.sample(label.to(device))
        blur = blur.detach().to('cpu')#.numpy()
        #blur = torch.from_numpy(blur) #2022/12/10 changed
        torch.save(blur, f'{outfolderpath}/{str(i+I).zfill(4)}_x{scale}.pt')

def save_label(folderpath, outfolderpath, labelname, outlabelname, I=0):
    flist = list(sorted(Path(folderpath).glob(f'*{labelname}.npy')))
    for i, label in enumerate(flist[I:]):
        label = torch.from_numpy(np.load(label))
        torch.save(label.float(),  f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt')
        
def create_mask_(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    mask = mask * 1.0
    return mask

def gen_jilist(model, model_name, val_dataset, device, partial=None):
    model.load_state_dict(torch.load(f'model/{model_name}.pt'))
    model.eval()
    jis = []
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach()
        image   = image.to(device=device).unsqueeze(0)
        pred, _ = model(image)
        pred    = pred.to(device='cpu').squeeze(0)
        if partial is not None:
            pred = pred[:, partial[0]:partial[1], :, :].detach()
        jis.append(jiffs(pred, label))
    return jis

def gen_bcelist(model, model_name, val_dataset, device, partial=None):
    model.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
    model.eval()
    bce = nn.BCELoss()
    bces = []
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach()
        image   = image.to(device=device).unsqueeze(0)
        pred, _ = model(image)
        pred    = pred.to(device='cpu').squeeze(0)
        if partial is not None:
            pred = pred[:, partial[0]:partial[1], :, :].detach()
        bces.append(bce(pred, label).to('cpu').item())
    return bces

def gen_bcecontrol(val_dataset, partial=None):
    bce = nn.BCELoss()
    bces = []
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach() * 1.0
        bces.append(bce(label, torch.ones_like(label) * torch.mean(label)).item())
    return bces

def torch_log2(x):
    return torch.clip(torch.log2(x), min=-100, max=100)

def bcelosswithlog2(inp, target):
    bcelg2 = -torch.mean(target * torch_log2(inp) + (1.0 - target) * torch_log2(1.0 - inp)).to('cpu')
    return bcelg2.item()

def gen_bcelg2list(model, model_name, val_dataset, device, partial=None):
    model.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
    model.eval()
    bces = []
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach()
        image   = image.to(device=device).unsqueeze(0)
        pred, _ = model(image)
        pred    = pred.to(device='cpu').squeeze(0)
        if partial is not None:
            pred = pred[:, partial[0]:partial[1], :, :].detach()
        bces.append(bcelosswithlog2(pred, label).to('cpu').item())
    return bces

def gen_bcelg2control(val_dataset, partial=None):
    bces = []
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        if partial is not None:
            label = label[:, partial[0]:partial[1], :, :].detach() * 1.0
        bces.append(bcelosswithlog2(torch.ones_like(label) * torch.mean(label), label).item())
    return bces

def gen_bcelg2lists_ctrls(model, model_names, val_datasets, device, taus, partials=[]):
    """
    returns bcess (0:control, 1~:model evals)
    """
    bcess = []
    ctrls = []
    for model_name, val_dataset, partial, tau in zip(model_names, val_datasets, partials, taus):
        model.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
        model.eval()
        model.set_tau(tau)
        bces = []
        ctrl = []
        for i in range(len(val_dataset)):
            image, label = val_dataset[i]
            if partial is not None:
                label = label[:, partial[0]:partial[1], :, :].detach()
            image   = image.to(device=device).unsqueeze(0)
            pred, _ = model(image)
            pred    = pred.to(device='cpu').squeeze(0)
            if partial is not None:
                pred = pred[:, partial[0]:partial[1], :, :].detach()
            bces.append(bcelosswithlog2(pred, label))
            ctrl.append(bcelosswithlog2(torch.ones_like(label) * torch.mean(label), label))
        bcess.append(bces)
        ctrls.append(ctrl)
    ctrl = [item for ctrl in ctrls for item in ctrl] # flatten control list
    bcess.insert(0, ctrl)
    return bcess

def mask_(image, mask_size, mask_num):
    """
    image     : 4d tensor
    mask_size : list with 3 elements (z, x, y)
    mask_num  : number of masks (default=1)
    out       : 4d tensor (randomly masked)
    """
    for i in range(mask_num):
        _c, _z, _x, _y = image.shape
        mask = torch.zeros((_c, *mask_size))
        _, mz, mx, my = mask.shape
        z = np.random.randint(0, _z)
        x = np.random.randint(0, _x)
        y = np.random.randint(0, _y)
        z_max = min(z + mz, _z)
        x_max = min(x + mx, _x)
        y_max = min(y + my, _y)
        image[:, z : z + mz, x : x + mx, y : y + my] \
        = mask[:, 0 : z_max - z, 0 : x_max - x, 0 : y_max - y]
    return image

def surround_mask_(image, surround_size):
    """
    image     : 4d tensor
    mask_size : list with 3 elements (z, x, y > 0)
    out       : 4d tensor (surround masked)
    """
    z, x, y = surround_size
    image = image[:, z : -z, x : -x, y : -y]
    image = F.pad(image, (y, y, x, x, z, z)) # see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    return image