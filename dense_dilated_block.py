import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvBlock, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_dep = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_point = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.ln = nn.LayerNorm(out_channels)
        self.activation = nn.PReLU()
    
    def forward(self, x):
        #x = self.conv(x)
        x = self.conv_dep(x)
        x = self.conv_point(x)
        # B x C x F x L -> B x F x L x C 
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        x = self.activation(x)
        return x 

class DilatedDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DilatedDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvBlock(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=2 ** i, dilation=2 ** i))
            
    def forward(self, x):
        out = x
        for i in range(self.num_layers):        
            out = torch.cat([out, self.layers[i](out)], 1)
        return out


if __name__=="__main__":
    net = DilatedDenseBlock(64, 4, 3)
    x = torch.rand([4, 64, 400, 79])
    print(net(x).shape)
