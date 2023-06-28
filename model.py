import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.checkpoint import checkpoint
from scipy.stats import lognorm
import time

from utils import tt


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

#class Attention(nn.Module):
#    def __init__(self):
#        self.to_a = 
#    def forward(self, x):
#
#        return x ## later

class BlurParameterEstimator(nn.Module):
    def __init__(self, x_dim, z_dim, mid_dim, params_dim):
        super().__init__()
        self.z_dim = z_dim
        self.to_a = nn.Sequential(nn.Linear(in_features  = x_dim,
                                            out_features = z_dim,),
                                  nn.Softmax(dim=-1))
        self.to_v = nn.Linear(in_features  = x_dim,
                              out_features = z_dim,)

        self.layers = nn.ModuleList([nn.Linear(in_features  = z_dim     ,
                                               out_features = mid_dim   ,),
                                     nn.ReLU(inplace=False)               ,
                                     nn.Linear(in_features  = mid_dim   ,
                                               out_features = mid_dim   ,),
                                     nn.ReLU(inplace=False)               ,
                                     nn.Linear(in_features  = mid_dim   ,
                                               out_features = params_dim,),
                                     ])
    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[1]) # (bczxy -> b'c) (b'=bzxy)
        a = self.to_a(x).view(x_shape[0], -1, self.z_dim) # (b'c -> b'c' -> bhc') (h=zxy)
        v = self.to_v(x).view(x_shape[0], -1, self.z_dim) # (b'c -> b'c' -> bhc') (h=zxy)
        z = torch.einsum("bhc, bhc -> bc", a, v)
        for f in self.layers:
            z = f(z)
        return z


class VectorQuantizer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    def forward(self, x):
        x_quantized = (x >= 0.5).to(self.device).float()
        x_quantized = x + (x_quantized - x).detach()
        quantize_loss = F.mse_loss(x_quantized.detach(), x)
        return x_quantized, quantize_loss


class JNetLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, param_estimation_list,
                 x_dim_list, params_dim, nblocks, dropout):
        super().__init__()
        self.last = hidden_channels_list[-1]
        hidden_channels = hidden_channels_list.pop(0)
        self.hidden_channels = hidden_channels
        self.is_param_estimation = param_estimation_list.pop(0)
        x_dim = x_dim_list.pop(0)
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
        self.param = BlurParameterEstimator(x_dim      = x_dim     ,
                                            z_dim      = 16        ,
                                            mid_dim    = 16        ,
                                            params_dim = params_dim,
                                            ) if self.is_param_estimation else nn.Identity()
        self.mid = JNetLayer(in_channels           = hidden_channels      ,
                             hidden_channels_list  = hidden_channels_list ,
                             param_estimation_list = param_estimation_list,
                             x_dim_list            = x_dim_list           ,
                             params_dim            = params_dim           ,
                             nblocks               = nblocks              ,
                             dropout               = dropout              ,
                             ) if hidden_channels_list else nn.Identity()
        self.post = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                             hidden_channels = hidden_channels,
                                             dropout         = dropout        ,
                                             ) for _ in range(nblocks)])
        self.unpool = JNetUnpooling(in_channels  = hidden_channels,
                                    out_channels = in_channels    ,)
    
    def forward(self, x):
        d = self.pool(x)
        d = checkpoint(self.conv, d) # checkpoint
        for f in self.prev:
            d = checkpoint(f, d)
        if self.hidden_channels == self.last:
            d = self.mid(d)
            p = self.param(d)
        else:
            d, p = self.mid(d)
        for f in self.post:
            d = checkpoint(f, d)
        d = checkpoint(self.unpool, d) # checkpoint
        x = x + d
        return x, p


class Emission(nn.Module):
    def __init__(self, mu_z, sig_z,):
        super().__init__()
        self.mu_z     = mu_z
        self.sig_z    = sig_z
        self.ez0      = nn.Parameter(torch.exp(mu_z + 0.5 * sig_z ** 2), requires_grad=True)
        #self.mu_z_    = mu_z.item()
        #self.sig_z_   = sig_z.item()
        #self.logn_ppf = lognorm.ppf([0.99], 1,
        #                    loc=self.mu_z_, scale=self.sig_z_)[0]
        
    def sample(self, x):
        b = x.shape[0]
        pz0  = dist.LogNormal(loc   = self.mu_z.view( b, 1, 1, 1, 1).expand(*x.shape),
                              scale = self.sig_z.view(b, 1, 1, 1, 1).expand(*x.shape),)
        x    = x * pz0.sample()
        x    = torch.clip(x, min=0, max=1)
        #x    = x / self.logn_ppf
        return x

    def forward(self, x):
        ez0 = torch.exp(self.mu_z + 0.5 * self.sig_z ** 2)
        x   = x * ez0
        x   = torch.clip(x, min=0, max=1)
        #x   = x / self.logn_ppf
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
    def __init__(self, z, x, y, bet_z, bet_xy, alpha, scale, device,
                 psf_mode:str="double_exp"):
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
        self.mode    = psf_mode
        self.psf     = self.gen_psf(self.bet_xy, self.bet_z, self.alpha
                                    ).to(device)
        self.z_pad   = (z - self.zscale + 1) // 2
        self.x_pad   = (x - self.xscale + 1) // 2
        self.y_pad   = (y - self.yscale + 1) // 2
        self.stride  = (self.zscale, self.xscale, self.yscale)
    
    def forward(self, x):
        x_shape = x.shape
        psf = self.gen_psf(self.bet_xy, self.bet_z, self.alpha
                           ).to(self.device)
        _x   = F.conv3d(input   = x                                    ,
                        weight  = psf                                  ,
                        stride  = self.stride                          ,
                        padding = (self.z_pad, self.x_pad, self.y_pad,),)
        batch = _x.shape[0]
        x = torch.empty(size=(*x_shape[:2], *_x.shape[2:])).to(self.device)
        for b in range(batch):
                x[b, 0] = _x[b, b]
        return x

    def gen_psf(self, bet_xy, bet_z, alpha):
        if bet_xy.shape:
            b = bet_xy.shape[0]
        else:
            b = 1
        psf_lateral = self.gen_2dnorm(self.dp, bet_xy, b)[:, None, None,    :,    :]
        psf_axial   = self.gen_1dnorm(self.zd, bet_z , b)[:, None,    :, None, None]
        psf  = torch.exp(torch.log(psf_lateral) + torch.log(psf_axial)) # log-sum-exp technique
        if self.mode == "gaussian":
            pass
        elif self.mode == "double_exp":
            if not alpha.shape:
                alpha = alpha.unsqueeze(0)
            psf  = self.gen_double_exp_dist(psf, alpha[:, None, None, None, None],)
        psf /= torch.sum(psf)
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

    def gen_2dnorm(self, distance_plane, bet_xy, b):
        distance_plane = distance_plane.expand(b, *distance_plane.shape)
        bet_xy = bet_xy.view(b, 1, 1).expand(*distance_plane.shape)
        d_2      =  distance_plane / bet_xy ** 2
        normterm = (torch.pi * 2) * (bet_xy ** 2)
        norm     = torch.exp(-d_2 / 2) / normterm
        return norm

    def gen_1dnorm(self, distance, bet_z, b):
        distance = distance.expand(b, *distance.shape)
        bet_z = bet_z.view(b, 1).expand(*distance.shape)
        d_2      =  distance ** 2 / bet_z ** 2
        normterm = (torch.pi * 2) ** 0.5 * bet_z
        norm     = torch.exp(-d_2 / 2) / normterm
        return norm

    def gen_double_exp_dist(self, norm, alpha,):
        pdf  = 1. - torch.exp(-alpha * norm)
        return pdf


class Noise(nn.Module):
    def __init__(self, sig_eps):
        super().__init__()
        self.sig_eps = sig_eps

    def forward(self, x):
        return x
    
    def sample(self, x):
        b = x.shape[0]
        px = dist.Normal(loc   = x           ,
                         scale = self.sig_eps.view(b, 1, 1, 1, 1).expand(*x.shape),)
        x  = px.rsample()
        x  = torch.clip(x, min=0, max=1)
        return x


class PreProcess(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        x = torch.clip(x, min=self.min, max=self.max)
        x = (x - self.min) / (self.max - self.min)
        return x


class ImagingProcess(nn.Module):
    def __init__(self, device, params,
                 z, x, y, postmin=0., postmax=1.,
                 mode:str="train",):
        super().__init__()
        self.device = device
        self.z = z
        self.x = x
        self.y = y
        self.postmin = postmin
        self.postmax = postmax
        if mode == "train":
            self.mu_z    = torch.tensor(params["mu_z"  ])
            self.sig_z   = torch.tensor(params["sig_z" ])
            self.bet_z   = nn.Parameter(torch.tensor(params["bet_z" ]), requires_grad=True)
            self.bet_xy  = nn.Parameter(torch.tensor(params["bet_xy"]), requires_grad=True)
            self.alpha   = nn.Parameter(torch.tensor(params["alpha" ]), requires_grad=True)
        elif mode == "dataset":
            self.mu_z    = nn.Parameter(torch.tensor(params["mu_z"  ]), requires_grad=False)
            self.sig_z   = nn.Parameter(torch.tensor(params["sig_z" ]), requires_grad=False)
            self.bet_z   = nn.Parameter(torch.tensor(params["bet_z" ]), requires_grad=False)
            self.bet_xy  = nn.Parameter(torch.tensor(params["bet_xy"]), requires_grad=False)
            self.alpha   = nn.Parameter(torch.tensor(params["alpha" ]), requires_grad=False)
        else:
            raise(NotImplementedError())
        scale = [params["scale"], 1, 1]
        #self.emission   = Emission(self.mu_z, self.sig_z)
        #self.blur       = Blur(z, x, y,
        #                       self.bet_z, self.bet_xy, self.alpha,
        #                       scale, device,)
        #self.noise      = Noise(params["sig_eps"])
        #self.preprocess = PreProcess(min=postmin, max=postmax)

    def forward(self, x):
        #x = self.emission(x) # rewrite to take (x, param)
        #x = self.blur(x)
        #x = self.preprocess(x)
        return x

    def sample(self, x):
        x = self.emission.sample(x)
        x = self.blur(x)
        x = self.noise.sample(x)
        x = self.preprocess(x)
        return x
    
    def sample_from_params(self, x, params):
        if isinstance(params["scale"], torch.Tensor):
            z = int(params["scale"][0])
        else:
            z = params["scale"]
        scale = [z, 1, 1] ## use only one scale
        emission   = Emission(tt(params["mu_z"]), tt(params["sig_z"]))
        blur       = Blur(self.z, self.x, self.y,
                          tt(params["bet_z"]), tt(params["bet_xy"]), tt(params["alpha"]),
                          scale, self.device)
        noise      = Noise(tt(params["sig_eps"]))
        preprocess = PreProcess(min=self.postmin, max=self.postmax)
        x = emission.sample(x)
        x = blur(x)
        x = noise.sample(x)
        x = preprocess(x)
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
                 dropout, params, param_estimation_list,
                 superres:bool, reconstruct=False, apply_vq=False,
                 device='cuda'):
        super().__init__()
        t1 = time.time()
        print('initializing model...')
        params_dim = int(len(params) - 2)
        scale_factor            = (params["scale"], 1, 1)
        hidden_channels_list    = hidden_channels_list.copy()
        hidden_channels = hidden_channels_list.pop(0)
        x_dim_list = hidden_channels_list.copy() 
        param_estimation = param_estimation_list.pop(0)
        self.prev0 = JNetBlock0(in_channels  = 1              ,
                                out_channels = hidden_channels,)
        self.prev  = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                              hidden_channels = hidden_channels,
                                              dropout         = dropout        ,
                                              ) for _ in range(nblocks)])
        self.mid   = JNetLayer(in_channels           = hidden_channels      ,
                               hidden_channels_list  = hidden_channels_list ,
                               param_estimation_list = param_estimation_list,
                               x_dim_list            = x_dim_list           ,
                               params_dim            = params_dim           ,
                               nblocks               = nblocks              ,
                               dropout               = dropout              ,
                               ) if hidden_channels_list else nn.Identity()
        self.post  = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                              hidden_channels = hidden_channels,
                                              dropout         = dropout        ,
                                              ) for _ in range(nblocks)])
        self.post0 = JNetBlockN(in_channels  = hidden_channels ,
                                out_channels = 2               ,)
        self.image = ImagingProcess(device       = device           ,
                                    params       = params           ,
                                    z            = 161              ,
                                    x            = 3                ,
                                    y            = 3                ,
                                    mode         = "train"          ,
                                    )
        self.upsample    = JNetUpsample(scale_factor = scale_factor)
        self.activation  = activation
        self.superres    = superres
        self.reconstruct = reconstruct
        self.apply_vq    = apply_vq
        self.vq = VectorQuantizer(device=device)
        t2 = time.time()
        print(f'init done ({t2-t1:.2f} s)')

    def set_upsample_rate(self, scale):
        scale_factor  = (scale, 1, 1)
        self.upsample = JNetUpsample(scale_factor = scale_factor)

    def forward(self, x):
        if self.superres:
            x = self.upsample(x)
        x = self.prev0(x)
        for f in self.prev:
            x = checkpoint(f, x)
        x, p = self.mid(x)
        for f in self.post:
            x = checkpoint(f, x)
        x = checkpoint(self.post0, x)
        x = F.softmax(input = x, dim = 1)[:, :1,] # softmax with temperature
        if self.apply_vq:
            x, qloss = self.vq(x)
        r = self.image(x) if self.reconstruct else x
        out = {"enhanced_image" : x,
               "reconstruction" : r,
               "blur_parameter" : p,
               }
        vqd = {"quantized_loss" : qloss} if self.apply_vq else {"quantized_loss" : None}
        out = dict(**out, **vqd)
        return out


if __name__ == '__main__':
    import torchinfo
    import torch.optim as optim
    surround = False
    surround_size = [32, 4, 4]
    hidden_channels_list = [16, 32, 64, 128, 256]
    nblocks              = 2
    s_nblocks            = 2
    activation           = nn.ReLU(inplace=True)
    dropout              = 0.5
    partial              = None #(56, 184)
    superres = True
    params               = {"mu_z"   : 0.2    ,
                            "sig_z"  : 0.2    ,
                            "bet_z"  : 23.5329,
                            "bet_xy" : 1.00000,
                            "alpha"  : 0.9544 ,
                            "sig_eps": 0.01   ,
                            "scale"  : 10
                            }
    image_size = (1, 1, 24, 96, 96)
    param_estimation_list = [False, False, False, False, True]
    model = JNet(hidden_channels_list  = hidden_channels_list ,
                 nblocks               = nblocks              ,
                 activation            = activation           ,
                 dropout               = dropout              ,
                 params                = params               ,
                 superres              = superres             ,
                 param_estimation_list = param_estimation_list,
                 image_size            = image_size           ,
                 reconstruct           = False                ,
                 apply_vq              = True                 ,
                 )
    input_size = (1, 1, 24, 96, 96)
    model.to(device='cuda')
    print(model(torch.abs(torch.randn(*input_size)).to(device='cuda')))
    #a, b, c, d, e = [i for i in model.parameters()][-5:]
    #print(a.item())
    torchinfo.summary(model, input_size)