import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet_Shallow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Shallow, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            DoubleConv(64, 32)
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder[0](enc)
        dec = self.decoder[1](torch.cat((dec, x), dim=1))
        return self.final_conv(dec)

class N2N_Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(N2N_Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        bottleneck = self.encoder[3](enc3)

        # Decoder
        dec1 = self.decoder[0](bottleneck)
        dec2 = self.decoder[1](dec1)
        dec3 = self.decoder[2](dec2)
        dec4 = self.decoder[3](dec3)
        dec5 = self.decoder[4](dec4)

        return self.final_conv(dec5)

class DNFLIM(nn.Module):
    def __init__(self):
        super(DNFLIM, self).__init__()
        self.branch1 = UNet_Shallow(in_channels=1, out_channels=32)
        self.branch2 = UNet_Shallow(in_channels=1, out_channels=32)
        self.denoise_net = N2N_Autoencoder(in_channels=64, out_channels=1)
        
    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        concatenated_output = torch.cat((branch1_output, branch2_output), dim=1)
        denoised_output = self.denoise_net(concatenated_output)
        return denoised_output


