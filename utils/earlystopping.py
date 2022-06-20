import numpy as np
import torch

class EarlyStopping():
    """
    modified from https://qiita.com/ku_a_i/items/ba33c9ce3449da23b503
    """
    def __init__(self, path, patience=10, verbose=False):
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.Inf
        self.path         = path
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss