import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.checkpoint import checkpoint
import torch.special as S
from scipy.stats import lognorm
from fft_conv_pytorch import fft_conv
import matplotlib.pyplot as plt
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


class CrossAttentionBlock(nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self, channels: int, n_heads: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        self.attn = CrossAttention(d_model = channels,
                                   d_cond  = d_cond,
                                   n_heads = n_heads,
                                   d_head  = channels // n_heads,)
        self.norm = nn.LayerNorm(normalized_shape = channels,)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).view(b, d * h * w, c)
        x = self.attn(self.norm(x)) + x
        x = x.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer
    This falls-back to self-attention when conditional embeddings are not specified.
    """

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads    = n_heads
        self.d_head     = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5
        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(in_features  = d_model ,
                              out_features = d_attn  ,
                              bias         = False   ,)
        
        self.to_k = nn.Linear(in_features  = d_cond  ,
                              out_features = d_attn  ,
                              bias         = False   ,)
        
        self.to_v = nn.Linear(in_features  = d_cond  ,
                              out_features = d_attn  ,
                              bias         = False   ,)
        # Final linear layer
        self.to_out = nn.Sequential(
                                        nn.Linear(in_features  = d_attn ,
                                                  out_features = d_model,),
                                    )

    def forward(self, x: torch.Tensor, cond=None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        has_cond = cond is not None
        if not has_cond:
            cond = x
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        return self.normal_attention(q, k, v)
    
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention
        :param q: `[batch_size, seq, d_attn]`
        :param k: `[batch_size, seq, d_attn]`
        :param v: `[batch_size, seq, d_attn]`
        """

        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, height * width * depth, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)

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
    def __init__(self, in_channels, hidden_channels_list,
                 attn_list, nblocks, dropout):
        super().__init__()
        is_attn = attn_list.pop(0)
        hidden_channels = hidden_channels_list.pop(0)
        self.hidden_channels = hidden_channels
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
        self.mid = JNetLayer(in_channels           = hidden_channels      ,
                             hidden_channels_list  = hidden_channels_list ,
                             attn_list             = attn_list            ,
                             nblocks               = nblocks              ,
                             dropout               = dropout              ,
                             ) if hidden_channels_list else nn.Identity()
        self.attn = CrossAttentionBlock(channels = hidden_channels ,
                                         n_heads  = 8               ,
                                         d_cond   = hidden_channels ,)\
                                             if is_attn else nn.Identity()
        self.post = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                             hidden_channels = hidden_channels,
                                             dropout         = dropout        ,
                                             ) for _ in range(nblocks)])
        self.unpool = JNetUnpooling(in_channels  = hidden_channels,
                                    out_channels = in_channels    ,)
    
    def forward(self, x):
        d = self.pool(x)
        d = self.conv(d) # checkpoint
        for f in self.prev:
            d = f(d)
        d = self.mid(d)
        d = self.attn(d)
        for f in self.post:
            d = f(d)
        d = self.unpool(d) # checkpoint
        x = x + d
        return x

class Emission(nn.Module):
    def __init__(self, mu_z, sig_z, log_ez0):
        super().__init__()
        self.mu_z     = mu_z
        self.sig_z    = sig_z
        self.log_ez0  = log_ez0

    def forward(self, x):
        x = x * torch.exp(self.log_ez0)
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
    def __init__(self, params, device, use_fftconv=False):
        super().__init__()
        self.params      = params
        self.device      = device
        self.use_fftconv = use_fftconv
        gibson_lanni     = GibsonLanniModel(params)
        self.psf_rz      = nn.Parameter(torch.tensor(gibson_lanni.PSF_rz, requires_grad=True).float().to(self.device))
        self.psf_rz_s0   = self.psf_rz.shape[0]
        xy               = torch.meshgrid(torch.arange(params["size_y"]),
                                          torch.arange(params["size_x"]),
                                          indexing='ij')
        #self.psf      = nn.Parameter(torch.zeros((self.params["size_z"], self.params["size_x"],  self.params["size_y"]), device=self.device), requires_grad=True)
        r = torch.tensor(gibson_lanni.r)
        x0 = (params["size_x"] - 1) / 2
        y0 = (params["size_y"] - 1) / 2
        r_pixel = torch.sqrt((xy[1] - x0) ** 2 + (xy[0] - y0) ** 2) * params["res_lateral"]
        rs0, = r.shape
        self.rps0, self.rps1 = r_pixel.shape
        r_e = r[:, None, None].expand(rs0, self.rps0, self.rps1)
        r_pixel_e = r_pixel[None].expand(rs0, self.rps0, self.rps1)
        r_index = torch.argmin(torch.abs(r_e- r_pixel_e), dim=0)
        r_index_fe = r_index.flatten().expand(self.psf_rz_s0, -1)
        self.r_index_fe = r_index_fe.to(self.device)
        self.z_pad   = int((params["size_z"] - self.params["res_axial"] // self.params["res_lateral"] + 1) // 2)
        self.x_pad   = (params["size_x"]) // 2
        self.y_pad   = (params["size_y"]) // 2
        self.stride  = (params["scale"], 1, 1)
        
    def forward(self, x):
        psf = torch.gather(self.psf_rz, 1, self.r_index_fe)
        psf = psf / torch.sum(psf)
        psf = psf.reshape(self.psf_rz_s0, self.rps0, self.rps1)
        if self.use_fftconv:
            _x   = fft_conv(signal  = x                                    ,
                            kernel  = psf                                  ,
                            stride  = self.stride                          ,
                            padding = (self.z_pad, self.x_pad, self.y_pad,),
                            )
        else:
            _x   = F.conv3d(input   = x                                    ,
                            weight  = psf                                  ,
                            stride  = self.stride                          ,
                            padding = (self.z_pad, self.x_pad, self.y_pad,),
                            )
        return _x

    def show_psf_3d(self):
        psf = torch.gather(self.psf_rz, 1, self.r_index_fe)
        psf = psf / torch.sum(psf)
        psf = psf.reshape(self.psf_rz_s0, self.rps0, self.rps1)
        return psf

class GibsonLanniModel():
    def __init__(self, params):
        size_x = params["size_x"]#256 # # # # param # # # #
        size_y = params["size_y"]#256 # # # # param # # # #
        size_z = params["size_z"]#128 # # # # param # # # #

        # Precision control
        num_basis    = 100  # Number of rescaled Bessels that approximate the phase function
        num_samples  = 1000 # Number of pupil samples along radial direction
        oversampling = 1    # Defines the upsampling ratio on the image space grid for computations

        # Microscope parameters
        NA          = params["NA"]        #1.1   # # # # param # # # #
        wavelength  = params["wavelength"]#0.910 # microns # # # # param # # # #
        M           = params["M"]         #25    # magnification # # # # param # # # #
        ns          = 1.33                       # specimen refractive index (RI)
        ng0         = 1.5                        # coverslip RI design value
        ng          = 1.5                        # coverslip RI experimental value
        ni0         = 1.5                        # immersion medium RI design value
        ni          = 1.5                        # immersion medium RI experimental value
        ti0         = 150                        # microns, working distance (immersion medium thickness) design value
        tg0         = 170                        # microns, coverslip thickness design value
        tg          = 170                        # microns, coverslip thickness experimental value
        res_lateral = params["res_lateral"]#0.05 # microns # # # # param # # # #
        res_axial   = params["res_axial"]#0.5    # microns # # # # param # # # #
        pZ          = 2                          # microns, particle distance from coverslip

        # Scaling factors for the Fourier-Bessel series expansion
        min_wavelength = 0.436 # microns
        scaling_factor = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wavelength / wavelength
        x0 = (size_x - 1) / 2
        y0 = (size_y - 1) / 2
        max_radius = round(np.sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0)))
        r = res_lateral * np.arange(0, oversampling * max_radius) / oversampling
        self.r = r
        a = min([NA, ns, ni, ni0, ng, ng0]) / NA
        rho = np.linspace(0, a, num_samples)

        z = res_axial * np.arange(-size_z / 2, size_z /2) + res_axial / 2

        OPDs = pZ * np.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample
        OPDi = (z.reshape(-1,1) + ti0) * np.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * np.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium
        OPDg = tg * np.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * np.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip
        W    = 2 * np.pi / wavelength * (OPDs + OPDi + OPDg)
        phase = np.cos(W) + 1j * np.sin(W)
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)
        C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T)
        b = 2 * np.pi * r.reshape(-1, 1) * NA / wavelength
        J0 = lambda x: scipy.special.j0(x)
        J1 = lambda x: scipy.special.j1(x)
        denom = scaling_factor * scaling_factor - b * b
        R = scaling_factor * J1(scaling_factor * a) * J0(b * a) * a - b * J0(scaling_factor * a) * J1(b * a) * a
        R /= denom
        PSF_rz = (np.abs(R.dot(C))**2).T
        self.PSF_rz = PSF_rz / np.max(PSF_rz)

    def __call__(self):
        return self.PSF_rz


class Noise(nn.Module):
    def __init__(self, sig_eps):
        super().__init__()
        self.sig_eps = sig_eps

    def forward(self, x):
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
                 mode:str="train", dist="double_exp",
                 apply_hill=False, use_fftconv=False):
        super().__init__()
        self.device = device
        self.z = z
        self.x = x
        self.y = y
        self.postmin = postmin
        self.postmax = postmax
        if mode == "train":
            self.mu_z       = params["mu_z"]
            self.sig_z      = params["sig_z"]
            self.log_ez0    = nn.Parameter((torch.tensor(params["mu_z"] + 0.5 * params["sig_z"] ** 2)).to(device), requires_grad=True)
        elif mode == "dataset":
            self.mu_z    = nn.Parameter(torch.tensor(params["mu_z"  ]), requires_grad=False)
            self.sig_z   = nn.Parameter(torch.tensor(params["sig_z" ]), requires_grad=False)
        else:
            raise(NotImplementedError())
        self.emission   = Emission(mu_z    = self.mu_z ,
                                   sig_z   = self.sig_z,
                                   log_ez0 = self.log_ez0,)
        self.blur       = Blur(params      = params         ,
                               device      = device         ,
                               use_fftconv = use_fftconv    ,)
        self.noise      = Noise(torch.tensor(params["sig_eps"]))
        self.preprocess = PreProcess(min=postmin, max=postmax)


    def forward(self, x):
        x = self.emission(x)
        x = self.blur(x)
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
    def __init__(self, hidden_channels_list, attn_list, nblocks, activation,
                 dropout, params, superres:bool, reconstruct=False,
                 apply_vq=False, use_fftconv=False,
                 use_x_quantized=True, blur_mode="gaussian", device='cuda',
                 z=161, x=3, y=3):
        super().__init__()
        t1 = time.time()
        print('initializing model...')
        scale_factor            = (params["scale"], 1, 1)
        hidden_channels_list    = hidden_channels_list.copy()
        hidden_channels = hidden_channels_list.pop(0)
        attn_list.pop(0)
        self.prev0 = JNetBlock0(in_channels  = 1              ,
                                out_channels = hidden_channels,)
        self.prev  = nn.ModuleList([JNetBlock(in_channels     = hidden_channels,
                                              hidden_channels = hidden_channels,
                                              dropout         = dropout        ,
                                              ) for _ in range(nblocks)])
        self.mid   = JNetLayer(in_channels           = hidden_channels      ,
                               hidden_channels_list  = hidden_channels_list ,
                               attn_list             = attn_list            ,
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
                                    z            = z                ,
                                    x            = x                ,
                                    y            = y                ,
                                    mode         = "train"          ,
                                    dist         = blur_mode        ,
                                    use_fftconv  = use_fftconv      ,
                                    )
        self.upsample    = JNetUpsample(scale_factor = scale_factor)
        self.activation  = activation
        self.superres    = superres
        self.reconstruct = reconstruct
        self.apply_vq    = apply_vq
        self.vq = VectorQuantizer(device=device)
        t2 = time.time()
        print(f'init done ({t2-t1:.2f} s)')
        self.use_x_quantized = use_x_quantized

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
        x = F.softmax(input = x, dim = 1)[:, :1,] # softmax with temperature
        if self.apply_vq:
            if self.use_x_quantized:
                x, qloss = self.vq(x)
            else:
                _, qloss = self.vq(x)
        r = self.image(x) if self.reconstruct else x
        out = {"enhanced_image" : x,
               "reconstruction" : r,
               }
        vqd = {"quantized_loss" : qloss} if self.apply_vq else {"quantized_loss" : None}
        out = dict(**out, **vqd)
        return out


if __name__ == '__main__':
    import torchinfo
    import torch.optim as optim
    surround = False
    surround_size = [32, 4, 4]
    hidden_channels_list = [4, 8, 16, 32]
    nblocks              = 2
    s_nblocks            = 2
    activation           = nn.ReLU(inplace=True)
    dropout              = 0.5
    partial              = None #(56, 184)
    superres = True
    params               = {"mu_z"       : 0.2    ,
                            "sig_z"      : 0.2    ,
                            "log_bet_z"  : 23.5329,
                            "log_bet_xy" : 1.00000,
                            "sig_eps"    : 0.01   ,
                            "scale"      : 10
                            }
    image_size = (4, 1, 24, 96, 96)
    model =  JNet(hidden_channels_list  = hidden_channels_list ,
                  attn_list=[False, False, False, True],
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  params                = params               ,
                  superres              = superres             ,
                  reconstruct           = True                 ,
                  apply_vq              = True                 ,
                  )
    input_size = (1, 1, 24, 96, 96)
    model.to(device='cuda')
    print([name for name, _ in model.named_parameters()])
    #a, b, c, d, e = [i for i in model.parameters()][-5:]
    #print(a.item())
    torchinfo.summary(model, input_size)