import torch
from torch.nn.functional import relu

class RCTDiffusionModel(torch.nn.Module):
    def __init__(self, denoise_levels):
        super().__init__()
        self.denoise_levels = denoise_levels

        # 256x256
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv1.weight)

        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv2.weight)

        self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv3.weight)

        # 64x64
        self.downsample1 = torch.nn.MaxPool2d(kernel_size=7, stride=4, padding=3)
        self.conv4 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv4.weight)

        self.conv5 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv5.weight)

        self.conv6 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv6.weight)

        # 16x16
        self.downsample2 = torch.nn.MaxPool2d(kernel_size=7, stride=4, padding=3)
        self.bilinear1 = torch.nn.Bilinear(in1_features=16 * 16 * 3, in2_features=1, out_features=16 * 16 * 3)
        self.bilinear2 = torch.nn.Bilinear(in1_features=16 * 16 * 3, in2_features=1, out_features=16 * 16 * 3)
        self.bilinear3 = torch.nn.Bilinear(in1_features=16 * 16 * 3, in2_features=1, out_features=16 * 16 * 3)

        self.conv7 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv7.weight)

        self.conv8 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv8.weight)

        self.conv9 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv9.weight)

        #64x64
        self.upsample1 = torch.nn.Upsample(scale_factor=4)

        self.conv10 = torch.nn.Conv2d(6, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv10.weight)

        self.conv11 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv11.weight)

        self.conv12 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv12.weight)

        # 256x256
        self.upsample2 = torch.nn.Upsample(scale_factor=4)

        self.conv13 = torch.nn.Conv2d(6, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv13.weight)

        self.conv14 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv14.weight)

        self.conv15 = torch.nn.Conv2d(3, 3, kernel_size=7, padding=3)
        torch.nn.init.xavier_normal_(self.conv15.weight)
    
    def forward(self, x, times):
        y0 = relu(self.conv1(x))
        y0 = relu(self.conv2(y0))
        y0 = relu(self.conv3(y0))

        y1 = self.downsample1(y0)
        y1 = relu(self.conv4(y1))
        y1 = relu(self.conv5(y1))
        y1 = relu(self.conv6(y1))

        y2 = self.downsample2(y1)
        y2_shape = y2.shape
        y2 = torch.flatten(y2, 1)
        y2 = relu(self.bilinear1(y2, times))
        y2 = relu(self.bilinear2(y2, times))
        y2 = relu(self.bilinear3(y2, times))
        y2 = torch.reshape(y2, y2_shape)
        y2 = relu(self.conv7(y2))
        y2 = relu(self.conv8(y2))
        y2 = relu(self.conv9(y2))

        y3 = self.upsample1(y2)

        y3 = torch.cat((y1, y3), dim=1)
        y3 = relu(self.conv10(y3))
        y3 = relu(self.conv11(y3))
        y3 = relu(self.conv12(y3))

        y4 = self.upsample2(y3)

        y4 = torch.cat((y0, y4), dim=1)
        y4 = relu(self.conv13(y4))
        y4 = relu(self.conv14(y4))
        y4 = relu(self.conv15(y4))
        return y4
