import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNetBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBranch, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        bottleneck = self.encoder[4](enc4)

        dec1 = self.decoder[0](bottleneck)
        dec2 = self.decoder[1](torch.cat((dec1, enc4), dim=1))
        dec3 = self.decoder[2](dec2)
        dec4 = self.decoder[3](torch.cat((dec3, enc3), dim=1))
        dec5 = self.decoder[4](dec4)
        dec6 = self.decoder[5](torch.cat((dec5, enc2), dim=1))
        dec7 = self.decoder[6](dec6)
        dec8 = self.decoder[7](torch.cat((dec7, enc1), dim=1))
        
        return self.final_conv(dec8)

class EncodeDecode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeDecode, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        bottleneck = self.encoder[3](enc3)

        dec1 = self.decoder[0](bottleneck)
        dec2 = self.decoder[1](torch.cat((dec1, enc3), dim=1))
        dec3 = self.decoder[2](dec2)
        dec4 = self.decoder[3](torch.cat((dec3, enc2), dim=1))
        dec5 = self.decoder[4](dec4)
        dec6 = self.decoder[5](torch.cat((dec5, enc1), dim=1))

        return self.final_conv(dec6)

class DNFLIM(nn.Module):
    def __init__(self):
        super(DNFLIM, self).__init__()
        self.branch1 = UNetBranch(in_channels=1, out_channels=64)
        self.branch2 = UNetBranch(in_channels=1, out_channels=64)
        self.denoise_net = EncodeDecode(in_channels=128, out_channels=1)
        
    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        concatenated_output = torch.cat((branch1_output, branch2_output), dim=1)
        denoised_output = self.denoise_net(concatenated_output)
        return denoised_output
