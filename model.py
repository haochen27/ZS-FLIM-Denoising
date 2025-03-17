import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Helper module: two consecutive conv layers with ReLU activations."""
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
    

class UpBlock(nn.Module):
    """
    Upsampling block using transposed convolution, followed by concatenation
    of skip connections and a double convolution.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels + skip_channels, out_channels)
    
    def forward(self, x, *skips):
        x = self.upconv(x)
        # Concatenate upsampled feature map with provided skip connections
        x = torch.cat((x, ) + skips, dim=1)
        return self.double_conv(x)
    

# -----------------------------------------------------------------------------
# Improved UNet with Shared Encoder for FLIM Training
# -----------------------------------------------------------------------------

class UNetSharedEncoder(nn.Module):
    """
    U-Net style network with two separate encoders (for x1 and x2) and three decoders:
      - Two decoders for reconstructing the individual inputs.
      - One combined decoder that fuses features from both encoders.
    """
    def __init__(self, in_channels=1, base_channels=32, out_channels=1):
        super(UNetSharedEncoder, self).__init__()
        # Encoder for first input (x1)
        self.enc1_1 = DoubleConv(in_channels, base_channels)
        self.pool1_1 = nn.MaxPool2d(2)
        self.enc1_2 = DoubleConv(base_channels, base_channels * 2)
        self.pool1_2 = nn.MaxPool2d(2)
        
        # Encoder for second input (x2)
        self.enc2_1 = DoubleConv(in_channels, base_channels)
        self.pool2_1 = nn.MaxPool2d(2)
        self.enc2_2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2_2 = nn.MaxPool2d(2)
        
        # Decoder for x1 branch
        self.up1_1 = UpBlock(base_channels * 2, skip_channels=base_channels * 2, out_channels=base_channels)
        self.up1_2 = UpBlock(base_channels, skip_channels=base_channels, out_channels=base_channels)
        self.final_conv1 = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Decoder for x2 branch
        self.up2_1 = UpBlock(base_channels * 2, skip_channels=base_channels * 2, out_channels=base_channels)
        self.up2_2 = UpBlock(base_channels, skip_channels=base_channels, out_channels=base_channels)
        self.final_conv2 = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Combined decoder (fusing both encoder streams)
        # The deepest features from both encoders are concatenated.
        self.up3_1 = UpBlock(base_channels * 4, skip_channels=base_channels * 4, out_channels=base_channels * 2)
        self.up3_2 = UpBlock(base_channels * 2, skip_channels=base_channels * 2, out_channels=base_channels)
        self.final_conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x1, x2):
        # ---- Encoding for x1 ----
        enc1_1 = self.enc1_1(x1)                    # [B, base_channels, H, W]
        enc1_2 = self.enc1_2(self.pool1_1(enc1_1))     # [B, base_channels*2, H/2, W/2]
        pooled1 = self.pool1_2(enc1_2)                # [B, base_channels*2, H/4, W/4]
        
        # ---- Encoding for x2 ----
        enc2_1 = self.enc2_1(x2)                    # [B, base_channels, H, W]
        enc2_2 = self.enc2_2(self.pool2_1(enc2_1))     # [B, base_channels*2, H/2, W/2]
        pooled2 = self.pool2_2(enc2_2)                # [B, base_channels*2, H/4, W/4]
        
        # ---- Decoder for x1 branch ----
        dec1_1 = self.up1_1(pooled1, enc1_2)
        dec1_2 = self.up1_2(dec1_1, enc1_1)
        out1 = self.final_conv1(dec1_2)
        
        # ---- Decoder for x2 branch ----
        dec2_1 = self.up2_1(pooled2, enc2_2)
        dec2_2 = self.up2_2(dec2_1, enc2_1)
        out2 = self.final_conv2(dec2_2)
        
        # ---- Combined Decoder ----
        # Fuse deepest features from both encoders.
        combined = torch.cat([pooled1, pooled2], dim=1)  # [B, base_channels*4, H/4, W/4]
        dec3_1 = self.up3_1(combined, enc1_2, enc2_2)
        dec3_2 = self.up3_2(dec3_1, enc1_1, enc2_1)
        out3 = self.final_conv3(dec3_2)
        
        return out1, out2, out3
 
class N2V_Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(N2V_Unet, self).__init__()
        # Encoder
        self.encoder1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)

        # Decoder with up-convolutions and skip connections
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

# (Optional) Residual and SE blocks can be integrated into your network if desired.
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

