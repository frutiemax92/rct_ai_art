import torch
from torch.nn.functional import relu

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, depth, in_channels=3, out_channels=3):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).to('cuda:0') for _ in range(depth)]) 
        self.conv_layers[0] = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to('cuda:0')
    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x
  
class LatentConvolutionBlock(torch.nn.Module):
    def __init__(self, depth, in_channels=3, mid_features=3, out_channels=3):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(in_channels, mid_features, kernel_size=3, padding=1).to('cuda:0') for _ in range(depth)]) 
        self.conv_layers[-1] = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to('cuda:0')
    def forward(self, x, times):
        for conv_layer in self.conv_layers:
            y = torch.cat([x, times], 1)
            x = conv_layer(y)
        return x

class BilinearBlock(torch.nn.Module):
    def __init__(self, depth, num_features):
        super().__init__()
        self.layer = torch.nn.ModuleList([torch.nn.Bilinear(num_features, num_features, num_features).to('cuda:0') for _ in range(depth)]) 

    def forward(self, x, times):
        for layer in self.layer:
            x = layer(x, times)
        return x

class LinearBlock(torch.nn.Module):
    def __init__(self, depth, in_features, out_features):
        super().__init__()
        self.layer = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features).to('cuda:0') for _ in range(depth)])
        self.layer[0] =  torch.nn.Linear(in_features, out_features).to('cuda:0')

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class RCTDiffusionModel(torch.nn.Module):
    denoise_levels = 32
    def __init__(self):
        super().__init__()

        conv_depth = 3
        features_base = 64
        # 256x256
        self.conv256 = ConvolutionBlock(conv_depth, 3, features_base).to('cuda:0')

        # downsampling 1
        self.downsample1 = torch.nn.MaxPool2d(kernel_size=3, stride=4)
        self.conv64 = ConvolutionBlock(conv_depth, features_base, features_base*2).to('cuda:0')

        # downsampling 2
        self.downsample2 = torch.nn.MaxPool2d(kernel_size=3, stride=8)
        self.latent_conv1 = ConvolutionBlock(conv_depth, features_base*2, features_base*4).to('cuda:0')
        self.time_linear_blocks_mult = LinearBlock(64, 1, 64*3)
        self.time_linear_blocks_add = LinearBlock(64, 1, 64*3)
        self.time_conv_block_add = ConvolutionBlock(conv_depth, 3, features_base*4)
        self.time_conv_block_mult = ConvolutionBlock(conv_depth, 3, features_base*4)
        self.latent_conv2 = LatentConvolutionBlock(conv_depth, features_base*8, features_base*4, features_base*2).to('cuda:0')

        #upsampling 1
        self.upsample1 = torch.nn.Upsample(scale_factor=8)
        self.conv64_out = ConvolutionBlock(conv_depth, features_base*4, features_base).to('cuda:0')

        # upsamping 2
        self.upsample2 = torch.nn.Upsample(scale_factor=4)
        self.conv256_out = ConvolutionBlock(conv_depth, features_base*2, 3).to('cuda:0')
    
    def forward(self, x, times):
        y0 = relu(self.conv256(x))

        y1 = self.downsample1(y0)
        y1 = relu(self.conv64(y1))

        y2 = self.downsample2(y1)
        y2 = relu(self.latent_conv1(y2))

        times_add = relu(self.time_linear_blocks_add(times))
        times_add = torch.reshape(times_add, (times.size(0), 3, 8, 8))
        times_add = relu(self.time_conv_block_add(times_add))

        times_mult = relu(self.time_linear_blocks_mult(times))
        times_mult = torch.reshape(times_mult, (times.size(0), 3, 8, 8))
        times_mult = relu(self.time_conv_block_mult(times_mult))

        y2_p = relu(y2 * times_mult + times_add)
        y2 = relu(self.latent_conv2(y2, y2_p))
        y3 = self.upsample1(y2)

        y3 = torch.cat((y1, y3), dim=1)
        y3 = relu(self.conv64_out(y3))

        y4 = self.upsample2(y3)

        y4 = torch.cat((y0, y4), dim=1)
        y4 = relu(self.conv256_out(y4))
        return y4
