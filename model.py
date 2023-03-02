import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from scipy.stats import lognorm
import time

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
    def __init__(self, scale_factor, z, x, y, mu_z, sig_z, bet_xy, bet_z, alpha,
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
        self.alpha   = nn.Parameter(torch.tensor(alpha  , requires_grad=True))
        self.logn_ppf = lognorm.ppf([0.99], 1, loc=mu_z, scale=sig_z)[0]
        self.zd,     \
        self.xd,     \
        self.yd      = self.distance(z, x, y, device)
        self.device  = device

    def distance(self, z, x, y, device):
        [zd, xd, yd] = [torch.zeros(1, 1, z, x, y, device=device) for _ in range(3)]
        for k in range(-z // 2, z // 2 + 1):
            zd[:, :, k + z // 2, :, :,] = k ** 2
        for i in range(-x // 2, x // 2 + 1):
            xd[:, :, :, i + x // 2, :,] = i ** 2
        for j in range(-y // 2, y // 2 + 1):
            yd[:, :, :, :, j + y // 2,] = j ** 2
        return zd, xd, yd

    def gen_alf(self, zd, xd, yd, bet_xy, bet_z, alpha):
        d_2 = zd / bet_z ** 2 + (xd + yd) / bet_xy ** 2
        alf = torch.exp(-d_2 / 2)
        normterm = (bet_z * bet_xy ** 2) * (torch.pi * 2) ** 1.5
        alf = alf / normterm 
        alf  = torch.ones_like(alf) - torch.exp(-alpha * alf)
        return alf

    def forward(self, inp):
        inp  = inp.unsqueeze(0) if inp.ndim == 4 else inp
        z0   = torch.exp(self.mu_z + 0.5 * self.sig_z ** 2) # E[z0|mu_z, sig_z]
        #z0   = z0 * torch.ones_like(inp, requires_grad=True)
        rec  = inp * z0 * 3.3
        #rec  = torch.clip(rec, min=0, max=self.logn_ppf)
        alf  = self.gen_alf(self.zd, self.xd, self.yd, self.bet_xy, self.bet_z, self.alpha)
        rec  = F.conv3d(input   = rec                               ,
                        weight  = alf                               ,
                        stride  = self.scale_factor                 ,
                        padding = ((self.z - self.zscale + 1) // 2  , 
                                   (self.x) // 2                    , 
                                   (self.y) // 2                    ,),)
        theorymax = self.logn_ppf * torch.sum(alf)
        rec  = rec / theorymax
        #rec  = (rec - rec.min()) / (rec.max() - rec.min())
        rec  = rec.squeeze(0) if inp.ndim == 4 else rec
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

class Emission(nn.Module):
    def __init__(self, mu_z, sig_z,):
        super().__init__()
        self.mu_z     = mu_z
        self.sig_z    = sig_z
        self.mu_z_    = mu_z.item()
        self.sig_z_   = sig_z.item()
        self.logn_ppf = lognorm.ppf([0.99], 1,
                            loc=self.mu_z_, scale=self.sig_z_)[0]
    def sample(self, x):
        pz0  = dist.LogNormal(loc   = self.mu_z  * torch.ones_like(x),
                              scale = self.sig_z * torch.ones_like(x),)    
        x    = x * pz0.sample()
        x    = torch.clip(x, min=0, max=self.logn_ppf)
        x    = x / self.logn_ppf
        return x

    def forward(self, x):
        ez0 = torch.exp(self.mu_z + 0.5 * self.sig_z ** 2)
        x   = x * ez0
        x   = torch.clip(x, min=0, max=self.logn_ppf)
        x   = x / self.logn_ppf
        return x

class Intensity(nn.Module):
    def __init__(self, gamma, image_size, initial_depth, voxel_size, scale,
                 device):
        super().__init__()
        init_d = initial_depth
        im_z,  \
        im_x,  \
        im_y   = image_size
        end_d  = im_z * voxel_size * scale
        depth  = init_d \
               + torch.linspace(0, end_d, im_z,).view(-1,1,1,).to(device)
        _intensity = torch.exp(-2 * depth * gamma) # gamma = 0.005
        self._intensity = _intensity.expand(image_size)

    def forward(self, x):
         x = x * self.intensity
         return x

    @property
    def intensity(self):
        return self._intensity


class Blur(nn.Module):
    def __init__(self, z, x, y, bet_z, bet_xy, alpha, scale, device,):
        super().__init__()
        self.bet_z   = bet_z
        self.bet_xy  = bet_xy
        self.alpha   = alpha
        self.zscale, \
        self.xscale, \
        self.yscale  = scale
        self.z       = z
        self.x       = x
        self.y       = y
        self.device  = device
        self.zd      = self.distance(z)
        self.dp      = self.gen_distance_plane(xlen=x, ylen=y)
        self.psf     = self.gen_psf(self.bet_xy, self.bet_z, self.alpha
                                    ).to(device)
        self.z_pad   = (z - self.zscale + 1) // 2
        self.x_pad   = (x - self.xscale + 1) // 2
        self.y_pad   = (y - self.yscale + 1) // 2
        self.stride  = (self.zscale, self.xscale, self.yscale)
    
    def forward(self, x):
        psf = self.gen_psf(self.bet_xy, self.bet_z, self.alpha
                           ).to(self.device)
        x   = F.conv3d(input   = x                                    ,
                       weight  = psf                                  ,
                       stride  = self.stride                          ,
                       padding = (self.z_pad, self.x_pad, self.y_pad,),)
        return x

    def gen_psf(self, bet_xy, bet_z, alpha):
        psf_lateral = self.gen_2dnorm(self.dp, bet_xy).view(1, self.x, self.y)
        psf_axial   = self.gen_1dnorm(self.zd, bet_z).view(self.z, 1, 1)
        psf  = torch.exp(torch.log(psf_lateral) + torch.log(psf_axial)) # log-sum-exp technique
        psf  = self.gen_double_exp_dist(psf, alpha,)
        psf /= torch.sum(psf)
        psf  = self.dim_3dto5d(psf)
        return psf

    def _init_distance(self, length):
        return torch.zeros(length)

    def _distance_from_center(self, index, length):
        return abs(index - length // 2)

    def distance(self, length):
        distance = torch.zeros(length)
        for idx in range(length):
            distance[idx] = self._distance_from_center(idx, length)
        return distance.to(self.device)

    def gen_distance_plane(self, xlen, ylen):
        xd = self.distance(xlen)
        yd = self.distance(ylen)
        xp = xd.expand(ylen, xlen)
        yp = yd.expand(xlen, ylen).transpose(1, 0)
        dp = xp ** 2 + yp ** 2
        return dp

    def gen_2dnorm(self, distance_plane, bet_xy):
        d_2      =  distance_plane / bet_xy ** 2
        normterm = (torch.pi * 2) * (bet_xy ** 2)
        norm     = torch.exp(-d_2 / 2) / normterm
        return norm

    def gen_1dnorm(self, distance, bet_z):
        d_2      =  distance ** 2 / bet_z ** 2
        normterm = (torch.pi * 2) ** 0.5 * bet_z
        norm     = torch.exp(-d_2 / 2) / normterm
        return norm

    def gen_double_exp_dist(self, norm, alpha,):
        pdf  = 1. - torch.exp(-alpha * norm)
        return pdf

    def dim_3dto5d(self, arr):
        return arr.view(1, 1, self.z, self.x, self.y)


class Noise(nn.Module):
    def __init__(self, sig_eps):
        super().__init__()
        self.sig_eps = sig_eps

    def foward(self, x):
        return x
    
    def sample(self, x):
        px = dist.Normal(loc   = x           ,
                         scale = self.sig_eps,)
        x  = px.rsample()
        x  = torch.clip(x, min=0, max=1)
        return x


class PreProcess(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        x = (torch.clip(x, min=self.min) - self.min) / (self.max - self.min)
        return x


class ImagingProcess(nn.Module):
    def __init__(self, mu_z, sig_z,
                 scale, device,
                 z, x, y, bet_z, bet_xy, alpha,
                 sig_eps, postmin=0.1, postmax=1,
                 gamma=0.05, image_size=(160, 128, 128),
                 initial_depth=0., voxel_size=0.05,
                 ):
        super().__init__()
        self.mu_z    = nn.Parameter(torch.tensor(mu_z  ), requires_grad=True)
        self.sig_z   = nn.Parameter(torch.tensor(sig_z ), requires_grad=True)
        self.bet_z   = nn.Parameter(torch.tensor(bet_z ), requires_grad=True)
        self.bet_xy  = nn.Parameter(torch.tensor(bet_xy), requires_grad=True)
        self.alpha   = nn.Parameter(torch.tensor(alpha ), requires_grad=True)
        self.emission   = Emission(self.mu_z, self.sig_z)
        #self.intensity  = Intensity(gamma, image_size, initial_depth,
        #                            voxel_size, scale[0], device,)
        self.blur       = Blur(z, x, y,
                               self.bet_z, self.bet_xy, self.alpha,
                               scale, device,)
        self.noise      = Noise(sig_eps)
        self.preprocess = PreProcess(min=postmin, max=postmax)

    def forward(self, x):
        x = self.emission(x)
        #x = self.intensity(x)
        x = self.blur(x)
        x = self.preprocess(x)
        return x

    def sample(self, x):
        x = self.emission.sample(x)
        #x = self.intensity(x)
        x = self.blur(x)
        x = self.noise.sample(x)
        x = self.preprocess(x)
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
                 mu_z:float, sig_z:float, bet_xy:float, bet_z:float, alpha:float,
                 superres:bool, reconstruct=False,device='cuda'):
        super().__init__()
        t1 = time.time()
        print('initializing model...')
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
        self.post0 = JNetBlockN(in_channels  = hidden_channels ,
                                out_channels = 2               ,)
        self.image = ImagingProcess(mu_z         = mu_z         ,
                                    sig_z        = sig_z        ,
                                    device       = device       ,
                                    scale        = scale_factor ,
                                    z            = 161          ,
                                    x            = 3           ,
                                    y            = 3           ,
                                    bet_xy       = bet_xy       ,
                                    bet_z        = bet_z        ,
                                    alpha        = alpha        ,
                                    sig_eps      = 0.01         ,
                                    postmin      = 0            ,
                                    postmax      = 1.           ,
                                    )
        self.upsample    = JNetUpsample(scale_factor = scale_factor)
        self.activation  = activation
        self.superres    = superres
        self.reconstruct = reconstruct
        t2 = time.time()
        print(f'init done ({t2-t1:.2f} s)')

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
        x = self.post0(x)
        x = F.softmax(input  = x / self.tau ,
                      dim    = 1            ,)[:, :1,] # softmax with temperature
        r = self.image(x) if self.reconstruct else x
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
                  alpha                 = 1.                   ,
                  superres              = True                 ,
                  reconstruct           = True                 ,
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