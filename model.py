import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.distributions as dist

class JNetBlock0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels  = in_channels ,
                              out_channels = out_channels,
                              kernel_size  = 7           ,
                              padding      = 'same'      ,
                              padding_mode = 'replicate' ,)
    def forward(self, x):
        x = self.conv(x)
        return x

class JNetBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.bn1      = nn.BatchNorm3d(num_features = in_channels)
        self.relu1    = nn.ReLU()
        self.conv1    = nn.Conv3d(in_channels  = in_channels    ,
                                  out_channels = hidden_channels,
                                  kernel_size  = 3              ,
                                  padding      = 'same'         ,
                                  padding_mode = 'replicate'    ,)
        
        self.bn2      = nn.BatchNorm3d(num_features = hidden_channels)
        self.relu2    = nn.ReLU()
        self.dropout1 = nn.Dropout(p = dropout)
        self.conv2    = nn.Conv3d(in_channels  = hidden_channels,
                                  out_channels = in_channels    ,
                                  kernel_size  = 3              ,
                                  padding      = 'same'         ,
                                  padding_mode = 'replicate'    ,)
    def forward(self, x):
        d = self.bn1(x)
        d = self.relu1(d)
        d = self.conv1(d)
        d = self.bn2(d)
        d = self.relu2(d)
        d = self.dropout1(d)
        d = self.conv2(d)
        x = x + d
        return x

class JNetBlockN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels  = in_channels ,
                              out_channels = out_channels,
                              kernel_size  = 3           ,
                              padding      = 'same'      ,
                              padding_mode = 'replicate' ,)
    def forward(self, x):
        x = self.conv(x)
        return x

class JNetPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size = 2)
        self.conv    = nn.Conv3d(in_channels  = in_channels    ,
                                 out_channels = out_channels   ,
                                 kernel_size  = 1              ,
                                 padding      = 'same'         ,
                                 padding_mode = 'replicate'    ,)
        self.relu    = nn.ReLU()
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class JNetUnpooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2            ,
                                    mode         = 'trilinear'  ,)
        self.conv     = nn.Conv3d(in_channels    = in_channels  ,
                                  out_channels   = out_channels ,
                                  kernel_size    = 1            ,
                                  padding        = 'same'       ,
                                  padding_mode   = 'replicate'  ,)
        self.relu     = nn.ReLU()
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class JNetSuperResolution(nn.Module):
    def __init__(self, scale_factor, in_channels, sr_hidden_channels_list, out_channels):
        super().__init__()
        sr_hidden_channels = sr_hidden_channels_list
        self.upsample = nn.Upsample(scale_factor = (scale_factor, 1, 1) ,
                                    mode         = 'trilinear'          ,)
        self.conv1    = nn.Conv3d(in_channels    = in_channels           ,
                                  out_channels   = sr_hidden_channels[0] ,
                                  kernel_size    = 9                     ,
                                  padding        = 'same'                ,
                                  padding_mode   = 'replicate'           ,)
        self.conv2    = nn.Conv3d(in_channels    = sr_hidden_channels[0] ,
                                  out_channels   = sr_hidden_channels[1] ,
                                  kernel_size    = 1                     ,
                                  padding        = 'same'                ,
                                  padding_mode   = 'replicate'           ,)
        self.conv3    = nn.Conv3d(in_channels    = sr_hidden_channels[1] ,
                                  out_channels   = out_channels          ,
                                  kernel_size    = 5                     ,
                                  padding        = 'same'                ,
                                  padding_mode   = 'replicate'           ,)
        self.relu     = nn.ReLU()
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

class JNetBlur(nn.Module):
    def __init__(self, scale, z, x, y, mu_z, sig_z, bet_xy, bet_z,):
        super().__init__()
        self.scale   = scale
        self.z       = z
        self.x       = x
        self.y       = y
        self.mu_z    = mu_z
        self.sig_z   = sig_z
        self.bet_xy  = bet_xy
        self.bet_z   = bet_z
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
        if inp.ndim == 4:
            inp  = inp.unsqueeze(0)

        z0   = torch.exp(self.mu_z + 0.5 * self.sig_z ** 2)
        z0   = z0 * torch.ones_like(inp)
        rec  = inp * z0
        rec  = F.conv3d(input   = rec                            ,
                        weight  = self.alf                       ,
                        stride  = (self.scale, 1, 1)             ,
                        padding = ((self.z - self.scale + 1) // 2, 
                                   (self.x) // 2                 , 
                                   (self.y) // 2                 ,))
        rec  = (rec - rec.min()) / (rec.max() - rec.min())
        #prec = dist.Normal(loc   = rec         ,
        #                   scale = self.sig_eps,)
        #rec  = prec.rsample()
        #rec  = (rec - rec.min()) / (rec.max() - rec.min())
        if inp.ndim == 4:
            rec  = rec.squeeze(0)
        return rec

class JNetLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, nblocks, dropout):
        super().__init__()
        hidden_channels = hidden_channels_list.pop(0)
        self.pool = JNetPooling(in_channels  = in_channels    ,
                                out_channels = hidden_channels,)
        self.conv = nn.Conv3d(in_channels    = hidden_channels,
                              out_channels   = hidden_channels,
                              kernel_size    = 1              ,
                              padding        = 'same'         ,
                              padding_mode   = 'replicate'    ,)
        self.prev = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                             hidden_channels = hidden_channels,
                                             dropout         = dropout        ,
                                             ) for _ in range(nblocks)])
        self.mid  = JNetLayer(in_channels          = hidden_channels     ,
                              hidden_channels_list = hidden_channels_list,
                              nblocks              = nblocks             ,
                              dropout              = dropout             ,
                              ) if hidden_channels_list else nn.Identity()
        self.post = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                             hidden_channels = hidden_channels,
                                             dropout         = dropout        ,
                                             ) for _ in range(nblocks)])
        self.unpool = JNetUnpooling(in_channels  = hidden_channels,
                                    out_channels = in_channels    ,)
    
    def forward(self, x):
        d = self.pool(x)
        d = self.conv(d)
        for f in self.prev:
            d = f(d)
        d = self.mid(d)
        for f in self.post:
            d = f(d)
        d = self.unpool(d)
        x = x + d
        return x

class JNet(nn.Module):
    def __init__(self, hidden_channels_list, nblocks, activation, dropout,
                 sr_hidden_channels_list, sr_out_channels, scale_factor, scale,
                 mu_z:float, sig_z:float, bet_xy:float, bet_z:float,):
        super().__init__()
        hidden_channels_list    = hidden_channels_list.copy()
        sr_hidden_channels_list = sr_hidden_channels_list.copy()
        hidden_channels         = hidden_channels_list.pop(0)
        self.prev0 = JNetBlock0(in_channels  = 1              ,
                                out_channels = hidden_channels,)
        self.prev  = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                              hidden_channels = hidden_channels,
                                              dropout         = dropout        ,
                                              ) for _ in range(nblocks)])
        self.mid   = JNetLayer(in_channels          = hidden_channels     ,
                               hidden_channels_list = hidden_channels_list,
                               nblocks              = nblocks             ,
                               dropout              = dropout             ,
                               ) if hidden_channels_list else nn.Identity()
        self.post  = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                              hidden_channels = hidden_channels,
                                              dropout         = dropout        ,
                                              ) for _ in range(nblocks)])

        self.sr    = JNetSuperResolution(scale_factor    = scale_factor       ,
                                         in_channels     = hidden_channels    ,
                                         sr_hidden_channels_list = sr_hidden_channels_list,
                                         out_channels    = sr_out_channels)
        self.post0 = JNetBlockN(in_channels  = sr_out_channels,
                                out_channels = 2              ,)
        #self.blur  = JNetBlur(scale = scale                                ,
        #                      z       = 141                                ,
        #                      x       = 7                                  ,
        #                      y       = 7                                  ,
        #                      mu_z    = nn.Parameter(torch.tensor(mu_z))   ,
        #                      sig_z   = nn.Parameter(torch.tensor(sig_z))  ,
        #                      bet_xy  = nn.Parameter(torch.tensor(bet_xy)) ,
        #                      bet_z   = nn.Parameter(torch.tensor(bet_z))  ,)

        self.activation = activation

    def forward(self, x):
        x = self.prev0(x)
        for f in self.prev:
            x = f(x)
        x = self.mid(x)
        for f in self.post:
            x = f(x)
        #print(x.shape)
        x = self.sr(x)
        x = self.post0(x)
        x = F.softmax(input  = x,
                      dim    = 1,)[:, :1,]

        #x = F.gumbel_softmax(logits = x   ,
        #                     tau    = 1.  ,
        #                     hard   = True, 
        #                     dim    = 1   ,)[:, :1,]
        #print('sr done', x.shape)
        #r = self.blur(x)
        r = 0
        #print('blur done',r.shape)
        return x, r