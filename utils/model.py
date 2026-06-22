import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_classes, upsample_mode='transpose'):
        super(SimpleUNet, self).__init__()
        # upsample_mode: 'transpose' (ConvTranspose2d, learnable) or
        # 'bilinear' (fixed interpolation, no channel reduction, no
        # checkerboard artifacts)
        assert upsample_mode in ('transpose', 'bilinear')
        self.upsample_mode = upsample_mode
 
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
 
        # Cuello de botella
        self.conv3 = DoubleConv(128, 256)
 
        # Decoder
        if upsample_mode == 'transpose':
            self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.conv4 = DoubleConv(256, 128) # 128 + 128 (skip connection)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.conv5 = DoubleConv(128, 64)
        else:
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv4 = DoubleConv(256 + 128, 128) # 256 (no channel reduction) + 128 (skip connection)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv5 = DoubleConv(128 + 64, 64)
 
        # Salida
        self.out_conv = nn.Conv2d(64, num_classes, 1)
 
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
 
        c3 = self.conv3(p2)
 
        u1 = self.up1(c3)
        # Skip connection 1
        u1 = torch.cat([u1, c2], dim=1)
        c4 = self.conv4(u1)
 
        u2 = self.up2(c4)
        # Skip connection 2
        u2 = torch.cat([u2, c1], dim=1)
        c5 = self.conv5(u2)
 
        out = self.out_conv(c5)
        return out
 
    
