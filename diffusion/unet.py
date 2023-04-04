# unet for diffusion
# num block variable
# add condition vec

import torch
import torch.nn as nn
import torch.nn.functional as F

# resblock (cond)
class ResBlock(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x, t, c):
        pass

# attentionblock
class Attention(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x, t, c):
        pass

# downblock
class DownBlock(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x, t, c):
        pass

# upblock
class UpBlock(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x, t, c):
        pass
    
# middleblock
class MiddleBlock(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x, t, c):
        pass