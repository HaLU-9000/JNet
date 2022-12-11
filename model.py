import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

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
        self.relu1    = nn.ReLU(inplace=True)
        self.conv1    = nn.Conv3d(in_channels  = in_channels    ,
                                  out_channels = hidden_channels,
                                  kernel_size  = 3              ,
                                  padding      = 'same'         ,
                                  padding_mode = 'replicate'    ,)
        
        self.bn2      = nn.BatchNorm3d(num_features = hidden_channels)
        self.relu2    = nn.ReLU(inplace=True)
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
        self.relu    = nn.ReLU(inplace=True)
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
        self.relu     = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class JNetUpsample(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor ,
                                    mode         = 'trilinear'  ,)
                                    
    def forward(self, x):
        return self.upsample(x)

class SuperResolutionBlock(nn.Module):
    def __init__(self, scale_factor, in_channels, nblocks, dropout):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor ,
                                    mode         = 'trilinear'  ,)
        self.post     = nn.ModuleList([JNetBlock(in_channels     = in_channels , 
                                                 hidden_channels = in_channels ,
                                                 dropout         = dropout     ,
                                                 ) for _ in range(nblocks)])
    def forward(self, x):
        x = self.upsample(x)
        for f in self.post:
            x = f(x)
        return x

class JNetBlur(nn.Module):
    def __init__(self, scale_factor, z, x, y, mu_z, sig_z, bet_xy, bet_z, 
                 device,):
        super().__init__()
        self.scale_factor = scale_factor
        self.zscale  = scale_factor[0]
        self.z       = z
        self.x       = x
        self.y       = y
        self.mu_z    = nn.Parameter(torch.tensor(mu_z   , requires_grad=True))
        self.sig_z   = nn.Parameter(torch.tensor(sig_z  , requires_grad=True))
        self.bet_xy  = nn.Parameter(torch.tensor(bet_xy , requires_grad=True))
        self.bet_z   = nn.Parameter(torch.tensor(bet_z  , requires_grad=True))

        self.zd, self.xd, self.yd   = self.distance(z, x, y, device)
        #self.alf     = self.gen_alf(zd, xd, yd, bet_xy, bet_z).to(device=device)
        self.device  = device
        #self.pz0     = dist.LogNormal(loc   = mu_z ,
        #                              scale = sig_z,)
    def distance(self, z, x, y, device):
        [zd, xd, yd] = [torch.zeros(1, 1, z, x, y, device=device) for _ in range(3)]
        for k in range(-z // 2, z // 2 + 1):
            zd[:, :, k + z // 2, :, :,] = k ** 2
        for i in range(-x // 2, x // 2 + 1):
            xd[:, :, :, i + x // 2, :,] = i ** 2
        for j in range(-y // 2, y // 2 + 1):
            yd[:, :, :, :, j + y // 2,] = j ** 2
        return zd, xd, yd

    def gen_alf(self, zd, xd, yd, bet_xy, bet_z):
        d_2 = zd / bet_z ** 2 + (xd + yd) / bet_xy ** 2
        alf = torch.exp(-d_2 / 2) / (torch.pi * 2) ** 0.5
        return alf

    def forward(self, inp):
        if inp.ndim == 4:
            inp  = inp.unsqueeze(0)
        #z0 = self.pz0.rsample(inp.shape)
        z0   = torch.exp(self.mu_z + 0.5 * self.sig_z ** 2) # E[z0|mu_z, sig_z]
        z0   = z0 * torch.ones_like(inp, requires_grad=True)
        rec  = inp * z0
        alf  = self.gen_alf(self.zd, self.xd, self.yd, self.bet_xy, self.bet_z)
        rec  = F.conv3d(input   = rec                               ,
                        weight  = alf                               ,
                        stride  = self.scale_factor                 ,
                        padding = ((self.z - self.zscale + 1) // 2, 
                                   (self.x) // 2                  , 
                                   (self.y) // 2                  ,),)
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

class SuperResolutionLayer(nn.Module):
    def __init__(self, in_channels, scale_list, nblocks, dropout):
        super().__init__()
        self.sr = nn.ModuleList([
                SuperResolutionBlock(scale_factor = scale_factor ,
                                     in_channels  = in_channels  ,
                                     nblocks      = nblocks      ,
                                    dropout      = dropout      ,
                                    )
                                 for scale_factor in scale_list])
    def forward(self, x):
        for f in self.sr:
            x = f(x)
        return x

class JNet(nn.Module):
    def __init__(self, hidden_channels_list, nblocks, activation,
                 dropout, scale_factor,
                 mu_z:float, sig_z:float, bet_xy:float, bet_z:float,
                 superres:bool, reconstruct=False,device='cuda'):
        super().__init__()
        hidden_channels_list    = hidden_channels_list.copy()
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
        # self.sr    = SuperResolutionLayer(in_channels   = hidden_channels ,
        #                                   scale_list    = scale_list      ,
        #                                   nblocks       = s_nblocks       ,
        #                                   dropout       = dropout         ,)
        self.post0 = JNetBlockN(in_channels  = hidden_channels ,
                                out_channels = 2               ,)
        self.blur  = JNetBlur(scale_factor = scale_factor ,
                              z            = 141          ,
                              x            = 35           ,
                              y            = 35           ,
                              mu_z         = mu_z         ,
                              sig_z        = sig_z        ,
                              bet_xy       = bet_xy       ,
                              bet_z        = bet_z        ,
                              device       = device       ,)
        self.upsample    = JNetUpsample(scale_factor = scale_factor)
        self.activation  = activation
        self.superres    = superres
        self.reconstruct = reconstruct
    def set_tau(self, tau=0.1):
        self.tau = tau
    def forward(self, x):
        if self.superres:
            x = self.upsample(x)
        x = self.prev0(x)
        for f in self.prev:
            x = f(x)
        x = self.mid(x)
        for f in self.post:
            x = f(x)
        # if self.superres:
        #     x = self.sr(x)
        x = self.post0(x)
        x = F.softmax(input  = x / self.tau ,
                      dim    = 1            ,)[:, :1,] # softmax with temperature
        if self.reconstruct:
            r = self.blur(x)
        else:
            r = x
        return x, r

if __name__ == '__main__':
    import torchinfo
    import torch.optim as optim
    hidden_channels_list = [16, 32, 64, 128, 256]
    scale_factor         = (10, 1, 1)
    nblocks              = 2
    activation           = nn.ReLU(inplace=True)
    dropout              = 0.5
    tau                  = 1.
    
    model =  JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_factor          = scale_factor         ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  ,
                  bet_xy                = 6.                   ,
                  bet_z                 = 35.                  ,
                  superres              = True                 ,
                  )
    model.set_tau(tau)
    input_size = (1, 1, 24, 112, 112)
    model.to(device='cuda')
    model.load_state_dict(torch.load('model/JNet_83_x1_partial.pt'), strict=False)
    model(torch.abs(torch.randn(*input_size)).to(device='cuda'))
    optimizer            = optim.Adam(model.parameters(), lr = 1e-4)

    a, b, c, d = [i for i in model.parameters()][-4:]
    print(a.item())
    torchinfo.summary(model, input_size)