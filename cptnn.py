import torch
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import sys, os 
# sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))


from TRANSFORMER import CPTB
from process_for_cptnn import seg_and_add_by_batch, restore_to_wav_by_batch
from dense_dilated_block import DilatedDenseBlock 
from random_mask import random_mask_by_batch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.ln = nn.LayerNorm(out_channels)
        self.activation = nn.PReLU()
    def forward(self, x):
        '''
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        '''
        f = self.conv(x)
        y = f.permute(0, 2, 3, 1).contiguous()
        y = self.ln(y)
        y = self.activation(y)
        z = y.permute(0, 3, 1, 2).contiguous()
        return z   


class Downsampler(nn.Module):
    def __init__(self, C=64, K=(1, 3), S=(1, 2), D=[2, 4, 8]):
        super(Downsampler, self).__init__()
        self.net = nn.ModuleList()
        for i in range(3):
            self.net.append(DilatedDenseBlock(C+i*8, 8, 1))
        self.conv = nn.Conv2d(C+3*8, C, K, S)
    def forward(self, x):
        for idx in range(len(self.net)):
            x = self.net[idx](x)
        x = self.conv(x)
        x = F.layer_norm(x, [x.shape[-1]])
        x = F.prelu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, first_channels=64, downsample_layer=2):
        super(Encoder, self).__init__()
        self.in_conv = ConvBlock(in_channels, first_channels, (1, 1), (1, 1))

        self.downsampler = nn.ModuleList()
        for i in range(downsample_layer):
            # self.downsampler.append(Downsampler(C=first_channels, K=(3, 1), S=(2, 1)))
            self.downsampler.append(Downsampler(C=first_channels, K=(1, 3), S=(1, 2)))
        self.out_conv = ConvBlock(first_channels, first_channels//2, (1, 1), (1, 1))        

    def forward(self, x):
        '''
        x: [batch, in_channels, num_frames, time_frames]
        return: [batch, first_channels, num_frames, time_frames]
        '''
        x = self.in_conv(x)
        for idx in range(len(self.downsampler)):
            x = self.downsampler[idx](x)
        f = self.out_conv(x)
        return x, f 


class CPTM(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_groups, cptm_layers=4):
        super(CPTM, self).__init__()
        self.layers = cptm_layers
        self.net = nn.ModuleList()
        for i in range(cptm_layers):
            self.net.append(CPTB(embed_dim, hidden_size, num_heads, num_groups))

    def forward(self, x):
        '''
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        '''        
        for i in range(self.layers):
            y = self.net[i](x)
            x = x + y 
        return x 

class MaskModule(nn.Module):
    def __init__(self, in_channels):
        super(MaskModule, self).__init__()
        self.up_conv = nn.Conv2d(in_channels, in_channels*2, (1, 1))
        self.activation1 = nn.PReLU()
        self.gated_conv = nn.Conv2d(in_channels*2, in_channels*2, (1, 1))
        self.activation2 = nn.Sigmoid()
        self.out_conv = nn.Conv2d(in_channels*2, in_channels*2, (1, 1))
        self.activation3 = nn.ReLU()

    def forward(self, x):
        '''
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        '''           
        x = self.activation1(self.up_conv(x))
        x = self.activation2(self.gated_conv(x))
        x = self.activation3(self.out_conv(x))
        return x

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, upscale_factor):
        super(Upsampler, self).__init__()
        self.dense_block = DilatedDenseBlock(in_channels, 8, 1)
        self.sub_pixel_conv = nn.Conv2d(out_channels+8, in_channels*(upscale_factor**2), kernel_size[1], stride[1], padding[1])
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    def forward(self, x):
        '''
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        '''
        x = self.dense_block(x)
        x = self.sub_pixel_conv(x)
        x = self.pixel_shuffle(x)
        x = F.layer_norm(x, [x.shape[-1]])
        x = F.prelu(x)        
        return x        

class Decoder(nn.Module):
    # def __init__(self, in_channels, hidden_size, kernel_size=[(2,2), (2,3)], stride=[(1,1), (1,2)], padding=[(0,0), (0,0)], dilation=[(1,1), (1,1)], out_channels=1, upsampler_layer=2):
    def __init__(self, in_channels, hidden_size, kernel_size=[(1,1), (3,1)], stride=[(1,1), (2,1)], padding=[(0,0), (0,0)], dilation=[(1,1), (1,1)], out_channels=1, upsampler_layer=2):
        super(Decoder, self).__init__()
        self.net = nn.ModuleList()
        for i in range(upsampler_layer):
            self.net.append(Upsampler(in_channels, hidden_size, kernel_size, stride, padding, dilation, upscale_factor=2))
        self.conv = nn.Conv2d(in_channels, out_channels, (1,1), padding=(2,2))

    def forward(self, x):
        for i in range(len(self.net)):
            x = self.net[i](x)
        return self.conv(x)


class CPTNN(nn.Module):
    def __init__(self,
                 frame_len=512,
                 hop_size=256, 
                 in_channels=1,
                 feat_dim=64,
                 downsample_layer=2,
                 hidden_size=64,
                 num_heads=4,
                 num_groups=4,
                 cptm_layers=4):
        super(CPTNN, self).__init__()
        self.frame_len = frame_len
        self.hop_size = hop_size

        self.encoder = Encoder(in_channels, feat_dim, downsample_layer)
        
        self.cptm = CPTM(feat_dim//2, hidden_size, num_heads, num_groups, cptm_layers)

        self.mask = MaskModule(feat_dim//2)

        self.decoder = Decoder(feat_dim, hidden_size)

    def forward(self, x):
        '''
        x: [batch, length]
        return: [batch, length]
        '''
        _, L = x.shape 
        x = seg_and_add_by_batch(x, self.frame_len, self.hop_size)
        x, f = self.encoder(x)
        # f = random_mask_by_batch(f)
        y = self.cptm(f)
        m = self.mask(y)
        x = x * m 
        x = self.decoder(x)
        x = restore_to_wav_by_batch(x, self.frame_len, self.hop_size)[...,:L]
        return x 


if __name__=="__main__":
    inputs = th.rand([4, 16000*4])
    print(inputs.shape)
    net = CPTNN()
    params = sum([param.nelement() for param in net.parameters()]) / 10.0**6
    print("params: {}M".format(params)) 
    outputs = net(inputs)
    print(outputs.shape)
