# unet for diffusion
# num block variable
# added condition vec (c)

import torch
import torch.nn as nn
import torch.nn.functional as F

# resblock (cond)
class ResBlock(nn.Module):
    def __init__(self,):
        
        # time embedding
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
class DownSample(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x, t, c):
        pass

# upblock
class UpSample(nn.Module):
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

# conbine
class Unet(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x, t, c):
        pass