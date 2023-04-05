# this code is modified from labml.ai https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SpatialTransformer


class UNetModel(nn.Module):
    """
    ## U-Net model
    """
    def __init__(
            self, *,
            in_channels: int,
            out_channels: int,
            channels: int,
            n_res_blocks: int,
            attention_levels: List[int],
            channel_multipliers: List[int],
            n_heads: int,
            tf_layers: int = 1,
            d_cond: int = 768):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: the number of attention heads in the transformers
        """
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)
        # Size time embeddings
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(in_features  = channels   ,
                      out_features = d_time_emb ,),
            nn.SiLU()                             ,
            nn.Linear(in_features  = d_time_emb ,
                      out_features = d_time_emb ,),
        )
        # Contraction Path of the U-Net
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            TimestepEmbedSequential(
                                    nn.Conv2d(in_channels  = in_channels ,
                                              out_channels = channels    ,
                                              kernel_size  = 3           ,
                                              padding      = 1           ,
                                              )
                                    )
                                )
        input_block_channels = [channels]
        channels_list        = [channels * m for m in channel_multipliers]
        for i in range(levels):
            for _ in range(n_res_blocks):
                layers   = [
                                ResBlock(channels    = channels        ,
                                         d_t_emb     = d_time_emb      ,
                                         out_channels= channels_list[i],)
                            ]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(channels = channels ,
                                           n_heads  = n_heads  ,
                                           n_layers = tf_layers,
                                           d_cond   = d_cond   ,
                                           )
                                  )
                self.input_blocks.append(
                    TimestepEmbedSequential(
                                            *layers
                                            )
                                        )
                input_block_channels.append(channels)
                
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                                            DownSample(
                                                       channels = channels
                                                       )
                                            )
                                         )
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                        channels = channels   ,
                        d_t_emb  = d_time_emb ,
                     )                                 ,
            SpatialTransformer(
                                channels = channels  ,
                                n_heads  = n_heads   ,
                                n_layers = tf_layers ,
                                d_cond   = d_cond    ,
                               )                       ,
            ResBlock(
                        channels = channels   ,
                        d_t_emb  = d_time_emb ,
                     )                                 ,
        )
        # extraction path (Second half) of the U-Net
        self.output_blocks = nn.ModuleList([])
        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [
                    ResBlock(channels     = channels 
                                            + input_block_channels.pop() ,
                             d_t_emb      = d_time_emb                   ,
                             out_channels = channels_list[i]             ,
                             )
                          ]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(
                                            channels = channels ,
                                            n_heads  = n_heads  ,
                                            n_layers = tf_layers,
                                            d_cond   = d_cond   ,
                                           )
                                  )
                if i != 0 and j == n_res_blocks:
                    layers.append(
                        UpSample(
                                    channels = channels
                                 )
                                  )
                # Add to the output half of the U-Net
                self.output_blocks.append(
                    TimestepEmbedSequential(
                                            *layers
                                            )
                                          )

        self.out = nn.Sequential(
                                  normalization(
                                                  channels = channels
                                                )                            ,
                                  nn.SiLU()                                  ,
                                  nn.Conv2d(
                                              in_channels  = channels     ,
                                              out_channels = out_channels ,
                                              kernel_size  = 3            ,
                                              padding      = 1            ,
                                            )                                ,
                                 )

    def time_step_embedding(
            self,
            time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        half        = self.channels // 2
        frequencies = torch.exp(
                                - math.log(max_period) 
                                * torch.arange(start = 0             ,
                                               end   = half          ,
                                               dtype = torch.float32 ,
                                               ) / half
                                ).to(device=time_steps.device)
        args        = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Get time step embeddings
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)
        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels  = channels,
                              out_channels = channels,
                              kernel_size  = 3       ,
                              padding      = 1       ,)
    
        self.up   = nn.Upsample(scale_factor = 2         ,
                                mode         = "nearest" ,)

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels  = channels ,
                              out_channels = channels ,
                              kernel_size  = 3        ,
                              stride       = 2        ,
                              padding      = 1        ,)

    def forward(self, x: torch.Tensor):
        x = self.down(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None,
                 activation=nn.SiLU(), dropout_rate=0.):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        if out_channels is None:
            out_channels = channels # `out_channels` not specified

        self.prev = nn.Sequential(
                                    normalization(channels)                 ,
                                    activation                              ,
                                    nn.Conv2d(in_channels  = channels     ,
                                              out_channels = out_channels ,
                                              kernel_size  = 3            ,
                                              stride       = 1            ,
                                              padding      = 1            ,),
                                 ) # First normalization and convolution
        
        self.emb  = nn.Sequential(
                                    activation,
                                    nn.Linear(d_t_emb,
                                              out_channels),
                                 ) # Time step embeddings

        self.post = nn.Sequential(
                                    normalization(out_channels)             ,
                                    activation                              ,
                                    nn.Dropout(p=dropout_rate)              ,
                                    nn.Conv2d(in_channels  = out_channels ,
                                              out_channels = out_channels ,
                                              kernel_size  = 3            ,
                                              stride       = 1            ,
                                              padding      = 1            ,),
                                 ) # Final convolution layer

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels  = channels    ,
                                             out_channels = out_channels,
                                             kernel_size  = 1           ,)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        h = self.prev(x)                   # Initial convolution
        t = self.emb(t).type(h.dtype)      # Time step embeddings
        h = h + t[:, :, None, None]        # Add time step embeddings
        h = self.post(h)                   # Final convolution
        return self.skip_connection(x) + h # Add skip connection


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization
    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(32, channels)


def _test_time_embeddings():
    """
    Test sinusoidal time step embeddings
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    m = UNetModel(in_channels=1, out_channels=1, channels=320, n_res_blocks=1, attention_levels=[],
                  channel_multipliers=[],
                  n_heads=1, tf_layers=1, d_cond=1)
    te = m.time_step_embedding(torch.arange(0, 1000))
    plt.plot(np.arange(1000), te[:, [50, 100, 190, 260]].numpy())
    plt.legend(["dim %d" % p for p in [50, 100, 190, 260]])
    plt.title("Time embeddings")
    plt.show()

if __name__ == "__main__":
    import torchinfo
    unet = UNetModel(in_channels=1,
                     out_channels=1,
                     channels=64,
                     n_res_blocks=2,
                     attention_levels=[1],
                     channel_multipliers=[1],
                     n_heads=8)
    torchinfo.summary(unet)