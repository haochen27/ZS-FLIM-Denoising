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
    
class UNet_SharedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_SharedEncoder, self).__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 16), 
            nn.MaxPool2d(2)  
        )
        
        # Decoder for the first input
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),  # Upsamples to match input size
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.001, inplace=True),
        )
        self.final_conv1 = nn.Conv2d(8, out_channels, kernel_size=1)
        
        # Decoder for the second input
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),  # Upsamples to match input size
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.001, inplace=True),
        )
        self.final_conv2 = nn.Conv2d(8, out_channels, kernel_size=1)
        
    def forward(self, x1, x2):
        # Shared encoding for both inputs
        enc1 = self.encoder(x1)  # Downsample and encode
        enc2 = self.encoder(x2)  # Downsample and encode (same encoder used)
        
        # Separate decoding for each input
        dec1 = self.decoder1(enc1)
        out1 = self.final_conv1(dec1)
        
        dec2 = self.decoder2(enc2)
        out2 = self.final_conv2(dec2)
        
        return out1, out2

class N2N_Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(N2N_Autoencoder, self).__init__()
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
            DoubleConv(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  
            DoubleConv(64, 64),
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0](x)  
        enc2 = self.encoder[1](enc1)  
        enc3 = self.encoder[2](enc2) 
        enc4 = self.encoder[3](enc3)  
        enc5 = self.encoder[4](enc4)  
        enc6 = self.encoder[5](enc5)  
        bottleneck = self.encoder[6](enc6)  

        # Decoder
        dec1 = self.decoder[0](bottleneck)  
        dec2 = self.decoder[1](dec1)  
        dec3 = self.decoder[2](dec2)  
        dec4 = self.decoder[3](dec3)  
        dec5 = self.decoder[4](dec4)  
        dec6 = self.decoder[5](dec5)
        output = self.final_conv(dec6)  

        return output


