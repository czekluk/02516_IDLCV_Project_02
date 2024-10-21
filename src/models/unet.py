import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDeconv(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.input_conv = nn.Conv2d(3, 64, 3, padding=1)  # 256 -> 256
        self.enc_conv0 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 128 -> 64
        self.enc_conv2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 64 -> 32
        self.enc_conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 32 -> 16

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(128, 128, 3, padding=1)  # 16 -> 16

        # Decoder (upsampling)
        self.trans_conv0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # 16 -> 32
        self.dec_conv0 = nn.Conv2d(256, 128, 3, padding=1)  # 128 from skip connection + 128 from upsampling

        self.trans_conv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # 32 -> 64
        self.dec_conv1 = nn.Conv2d(256, 128, 3, padding=1)  # 128 from skip connection + 128 from upsampling

        self.trans_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # 64 -> 128
        self.dec_conv2 = nn.Conv2d(192, 128, 3, padding=1)  # 128 from upsampling + 64 from skip connection

        self.trans_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128 -> 256
        self.dec_conv3 = nn.Conv2d(128, 64, 3, padding=1)  # 64 from upsampling + 64 from skip connection

        # Output layer
        self.output_conv = nn.Conv2d(64, 1, 1)  # Final output should have a single channel

    def forward(self, x):
        # Encoder
        input = F.relu(self.input_conv(x))  # 256
        e0 = F.relu(self.enc_conv0(input))  # 128
        e1 = F.relu(self.enc_conv1(e0))  # 64
        e2 = F.relu(self.enc_conv2(e1))  # 32
        e3 = F.relu(self.enc_conv3(e2))  # 16

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3))  # 16

        # Decoder
        d0_upsampled = F.relu(self.trans_conv0(b))  # 16 -> 32
        d0 = F.relu(self.dec_conv0(torch.cat([d0_upsampled, e2], dim=1)))  # 256 channels total (128+128)

        d1_upsampled = F.relu(self.trans_conv1(d0))  # 32 -> 64
        d1 = F.relu(self.dec_conv1(torch.cat([d1_upsampled, e1], dim=1)))  # 256 channels total (128+128)

        d2_upsampled = F.relu(self.trans_conv2(d1))  # 64 -> 128
        d2 = F.relu(self.dec_conv2(torch.cat([d2_upsampled, e0], dim=1)))  # 192 channels total (128+64)

        d3_upsampled = F.relu(self.trans_conv3(d2))  # 128 -> 256
        d3 = F.relu(self.dec_conv3(torch.cat([d3_upsampled, input], dim=1)))  # 128 channels total (64+64)

        # Output layer
        output = self.output_conv(d3)  # Final output with a single channel

        return output

    
class UNetDilated(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (using dilated convolutions)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1, dilation=1)  # Regular conv, 256x256 -> 256x256
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)  # Dilated conv, 256x256 -> 256x256
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)  # Dilated conv, 256x256 -> 256x256
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)  # Dilated conv, 256x256 -> 256x256
        self.enc_conv4 = nn.Conv2d(64, 64, 3, padding=16, dilation=16)  # 256x256 -> 256x256

        # Bottleneck (dilated convolution with a large receptive field)
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=32, dilation=32)  # 256x256 -> 256x256

        # Decoder (reducing dilation gradually)
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=16, dilation=16)  # 256x256 -> 256x256
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)  # 256x256 -> 256x256
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)  # 256x256 -> 256x256
        self.dec_conv3 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)  # 256x256 -> 256x256
        self.dec_conv4 = nn.Conv2d(64, 1, 3, padding=1, dilation=1)   # 256x256 -> 256x256

    def forward(self, x):
        # Encoder (with dilated convolutions)
        e0 = F.relu(self.enc_conv0(x))  # 256x256 input remains 256x256
        e1 = F.relu(self.enc_conv1(e0))  # Dilation=2, 256x256
        e2 = F.relu(self.enc_conv2(e1))  # Dilation=4, 256x256
        e3 = F.relu(self.enc_conv3(e2))  # Dilation=8, 256x256
        e4 = F.relu(self.enc_conv3(e3))  # Dilation=8, 256x256


        # Bottleneck (large receptive field)
        b = F.relu(self.bottleneck_conv(e4))  # Dilation=16, 256x256

        # Decoder (gradually reducing dilation)
        d0 = F.relu(self.dec_conv0(b))  # Dilation=8, 256x256
        d1 = F.relu(self.dec_conv1(d0))  # Dilation=4, 256x256
        d2 = F.relu(self.dec_conv2(d1))  # Dilation=2, 256x256
        d3 = F.relu(self.dec_conv3(d2))  # Dilation=2, 256x256
        d4 = self.dec_conv4(d3)  # Regular convolution, 256x256

        return d4
    