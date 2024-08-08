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
        
        self.encoder1 = nn.Sequential(
            DoubleConv(in_channels, 16), 
            nn.MaxPool2d(2)
        )
        
        # Encoder for the second input
        self.encoder2 = nn.Sequential(
            DoubleConv(in_channels, 16), 
            nn.MaxPool2d(2)
        )
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
        
        # Decoder for the concatenated features of enc1 and enc2
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # Upsamples to match input size
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.001, inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.001, inplace=True),
        )
        self.final_conv3 = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # Shared encoding for both inputs
        enc1 = self.encoder1(x1)  # Downsample and encode
        enc2 = self.encoder2(x2)  # Downsample and encode (same encoder used)
        
        # Separate decoding for each input
        dec1 = self.decoder1(enc1)
        out1 = self.final_conv1(dec1)
        
        dec2 = self.decoder2(enc2)
        out2 = self.final_conv2(dec2)
        
        # Concatenate encoded features and decode
        concat_features = torch.cat((enc1, enc2), dim=1)  # Concatenate along channel dimension
        dec3 = self.decoder3(concat_features)
        out3 = self.final_conv3(dec3)
        
        return out1, out2, out3

class N2V_Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(N2V_Unet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(256, 512)

        self.decoder1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv1 = DoubleConv(512, 256)
        self.decoder2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv2 = DoubleConv(256, 128)
        self.decoder3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv3 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.encoder4(self.pool3(enc3))

        # Decoder with skip connections
        dec1 = self.decoder1(bottleneck)
        dec1 = self.decoder_conv1(torch.cat([dec1, enc3], dim=1))
        dec2 = self.decoder2(dec1)
        dec2 = self.decoder_conv2(torch.cat([dec2, enc2], dim=1))
        dec3 = self.decoder3(dec2)
        dec3 = self.decoder_conv3(torch.cat([dec3, enc1], dim=1))

        output = self.final_conv(dec3)

        return output

class FrequencyDomainLoss(nn.Module):
    def __init__(self, mask_size=10):
        super(FrequencyDomainLoss, self).__init__()
        self.mask_size = mask_size

    def forward(self, img1, img2):
        # Ensure images are the same size
        assert img1.shape == img2.shape, "Images must have the same shape"

        # Compute FFT of the images
        fft_img1 = torch.fft.fft2(img1)
        fft_img2 = torch.fft.fft2(img2)

        # Shift the zero frequency component to the center
        fft_img1 = torch.fft.fftshift(fft_img1)
        fft_img2 = torch.fft.fftshift(fft_img2)

        # Create a low-frequency mask
        mask = self.create_low_frequency_mask(fft_img1.shape, self.mask_size)
        mask = mask.to(img1.device)

        # Apply the mask to the frequency domain representations
        masked_fft_img1 = fft_img1 * mask
        masked_fft_img2 = fft_img2 * mask

        # Compute the inverse FFT to get back to the spatial domain
        ifft_img1 = torch.fft.ifftshift(masked_fft_img1)
        ifft_img2 = torch.fft.ifftshift(masked_fft_img2)
        spatial_img1 = torch.fft.ifft2(ifft_img1).real
        spatial_img2 = torch.fft.ifft2(ifft_img2).real

        # Compute the loss (e.g., L2 loss) between the low-frequency components
        loss = nn.MSELoss()(spatial_img1, spatial_img2)
        return loss
    
    def create_low_frequency_mask(self, shape, size):
        mask = torch.zeros(shape)
        center = [s // 2 for s in shape[-2:]]
        for i in range(center[0] - size, center[0] + size):
            for j in range(center[1] - size, center[1] + size):
                mask[..., i, j] = 1
        return mask
