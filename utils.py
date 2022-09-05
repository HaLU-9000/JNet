import numpy as np
import torch
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
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:
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

def save_dataset(model, folderpath, outfolderpath, labelname, outlabelname, scale, I=0):
    flist = list(sorted(Path(folderpath).glob(f'*{labelname}.npy')))
    for i, label in enumerate(flist[I:]):
        label = torch.from_numpy(np.load(label))
        if not Path(f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt').is_file():
            torch.save(label.float(),  f'{outfolderpath}/{str(i+I).zfill(4)}{outlabelname}.pt')
        blur = model(label)
        blur = blur.detach().to('cpu').numpy()
        blur = torch.from_numpy(blur)
        torch.save(blur, f'{outfolderpath}/{str(i+I).zfill(4)}_x{scale}.pt')

def create_mask(h, w, center=None, radius=None):
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