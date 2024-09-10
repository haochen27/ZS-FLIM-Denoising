import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Separate encoders for both inputs
        # Encoder for x1
        self.encoder1_1 = DoubleConv(in_channels, 32)
        self.pool1_1 = nn.MaxPool2d(2)
        self.encoder1_2 = DoubleConv(32, 64)
        self.pool1_2 = nn.MaxPool2d(2)

        # Encoder for x2
        self.encoder2_1 = DoubleConv(in_channels, 32)
        self.pool2_1 = nn.MaxPool2d(2)
        self.encoder2_2 = DoubleConv(32, 64)
        self.pool2_2 = nn.MaxPool2d(2)
        
        # Two-level U-Net style decoders for the first and second input
        # Decoder for out1
        self.upconv1_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1_1 = DoubleConv(96, 32)
        self.upconv1_2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.decoder1_2 = DoubleConv(64, 32)
        self.final_conv1 = nn.Conv2d(32, out_channels, kernel_size=1)

        # Decoder for out2
        self.upconv2_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2_1 = DoubleConv(96, 32)
        self.upconv2_2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.decoder2_2 = DoubleConv(64, 32)
        self.final_conv2 = nn.Conv2d(32, out_channels, kernel_size=1)

        # New decoder for out3, using features from both enc1 and enc2 paths
        self.upconv3_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsampling combined features
        self.decoder3_1 = DoubleConv(192, 64)
        self.upconv3_2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # Second level upsampling
        self.decoder3_2 = DoubleConv(96, 32)
        self.final_conv3 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # Encoding path for x1
        enc1_1 = self.encoder1_1(x1)                     # Output: [batch_size, 32, 256, 256]
        enc1_pooled1 = self.pool1_1(enc1_1)              # Output: [batch_size, 32, 128, 128]
        enc1_2 = self.encoder1_2(enc1_pooled1)           # Output: [batch_size, 64, 128, 128]
        enc1_pooled2 = self.pool1_2(enc1_2)              # Output: [batch_size, 64, 64, 64]

        # Encoding path for x2
        enc2_1 = self.encoder2_1(x2)                     # Output: [batch_size, 32, 256, 256]
        enc2_pooled1 = self.pool2_1(enc2_1)              # Output: [batch_size, 32, 128, 128]
        enc2_2 = self.encoder2_2(enc2_pooled1)           # Output: [batch_size, 64, 128, 128]
        enc2_pooled2 = self.pool2_2(enc2_2)              # Output: [batch_size, 64, 64, 64]

        # U-Net style decoding for the first input (out1)
        upconv1 = self.upconv1_1(enc1_pooled2)            # Output: [batch_size, 32, 128, 128]
        dec1 = self.decoder1_1(torch.cat([upconv1, enc1_2], dim=1)) # Skip connection with enc1_2
        upconv1 = self.upconv1_2(dec1)                    # Output: [batch_size, 32, 256, 256]
        dec1 = self.decoder1_2(torch.cat([upconv1, enc1_1], dim=1)) # Skip connection with enc1_1
        out1 = self.final_conv1(dec1)                     # Output: [batch_size, out_channels, 256, 256]

        # U-Net style decoding for the second input (out2)
        upconv2 = self.upconv2_1(enc2_pooled2)            # Output: [batch_size, 32, 128, 128]
        dec2 = self.decoder2_1(torch.cat([upconv2, enc2_2], dim=1)) # Skip connection with enc2_2
        upconv2 = self.upconv2_2(dec2)                    # Output: [batch_size, 32, 256, 256]
        dec2 = self.decoder2_2(torch.cat([upconv2, enc2_1], dim=1)) # Skip connection with enc2_1
        out2 = self.final_conv2(dec2)                     # Output: [batch_size, out_channels, 256, 256]

        # New U-Net style decoding for the combined features (out3)
        combined_features = torch.cat([enc1_pooled2, enc2_pooled2], dim=1)  # Combine the deepest features [batch_size, 128, 64, 64]
        upconv3 = self.upconv3_1(combined_features)                         # Upsample to [batch_size, 64, 128, 128]
        dec3 = self.decoder3_1(torch.cat([upconv3, enc1_2, enc2_2], dim=1)) # Skip connection with enc1_2 and enc2_2
        upconv3 = self.upconv3_2(dec3)                                      # Upsample to [batch_size, 32, 256, 256]
        dec3 = self.decoder3_2(torch.cat([upconv3, enc1_1, enc2_1], dim=1)) # Skip connection with enc1_1 and enc2_1
        out3 = self.final_conv3(dec3)                                       # Output: [batch_size, out_channels, 256, 256]

        return out1, out2, out3

class N2V_Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(N2V_Unet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))

        output = self.final_conv(dec1)

        return output
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.global_avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        scale = self.sigmoid(out)
        return x * scale

