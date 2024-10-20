import torch
import torch.nn as nn
import torch.nn.functional as F

class EncDec_base(nn.Module):
    """Encoder decoder structure"""
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 1024, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv3 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv4 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(1024, 1024, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')  # 128 -> 256
        self.dec_conv4 = nn.Conv2d(1024, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))
        e4 = self.pool4(F.relu(self.enc_conv4(e3)))
        # bottleneck
        b = F.relu(self.bottleneck_conv(e4))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = F.relu(self.dec_conv3(self.upsample3(d2)))
        d4 = self.dec_conv4(self.upsample4(d3))  # no activation
        return d4


class EncDecStride(nn.Module):
    """Encoder-decoder structure with stride 2 for downsampling and ConvTranspose2d for upsampling"""
    def __init__(self):
        super().__init__()

        # encoder (downsampling with stride 2)
        self.enc_conv0 = nn.Conv2d(3, 1024, 3, padding=1, stride=2)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)  # 128 -> 64
        self.enc_conv2 = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)  # 64 -> 32
        self.enc_conv3 = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)  # 32 -> 16
        self.enc_conv4 = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(1024, 1024, 3, padding=1)

        # decoder (upsampling with ConvTranspose2d)
        self.dec_conv0 = nn.ConvTranspose2d(1024, 1024, 3, padding=1, stride=2, output_padding=1)  # 8 -> 16
        self.dec_conv1 = nn.ConvTranspose2d(1024, 1024, 3, padding=1, stride=2, output_padding=1)  # 16 -> 32
        self.dec_conv2 = nn.ConvTranspose2d(1024, 1024, 3, padding=1, stride=2, output_padding=1)  # 32 -> 64
        self.dec_conv3 = nn.ConvTranspose2d(1024, 1024, 3, padding=1, stride=2, output_padding=1)  # 64 -> 128
        self.dec_conv4 = nn.ConvTranspose2d(1024, 1, 3, padding=1, stride=2, output_padding=1)  # 128 -> 256

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))
        e3 = F.relu(self.enc_conv3(e2))
        e4 = F.relu(self.enc_conv4(e3))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e4))

        # decoder
        d0 = F.relu(self.dec_conv0(b))
        d1 = F.relu(self.dec_conv1(d0))
        d2 = F.relu(self.dec_conv2(d1))
        d3 = F.relu(self.dec_conv3(d2))
        d4 = self.dec_conv4(d3)  # no activation
        return d4

class EncDec_dropout(EncDec_base):
    """Encoder decoder structure with dropout"""
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))
        e4 = self.pool4(F.relu(self.enc_conv4(e3)))
        # bottleneck
        b = F.relu(self.bottleneck_conv(e4))
        b = self.dropout(b)

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d0 = self.dropout(d0)
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d1 = self.dropout(d1)
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d2 = self.dropout(d2)
        d3 = F.relu(self.dec_conv3(self.upsample3(d2)))
        d3 = self.dropout(d3)
        d4 = self.dec_conv4(self.upsample4(d3))  # no activation
        return d4


class DilatedConvNet(nn.Module):
    """Network with only dilated convolutions and increasing dilation rates"""
    def __init__(self):
        super().__init__()
        # Dilated convolutions with increasing dilation rates
        self.dilated_conv0 = nn.Conv2d(3, 256, 3, padding=1, dilation=1)
        self.dilated_conv1 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(256, 256, 3, padding=4, dilation=4)
        self.dilated_conv3 = nn.Conv2d(256, 256, 3, padding=8, dilation=8)
        self.dilated_conv4 = nn.Conv2d(256, 256, 3, padding=16, dilation=16)
        self.dilated_conv5 = nn.Conv2d(256, 1, 3, padding=1)  # Output layer

    def forward(self, x):
        # Apply dilated convolutions
        x = F.relu(self.dilated_conv0(x))
        x = F.relu(self.dilated_conv1(x))
        x = F.relu(self.dilated_conv2(x))
        x = F.relu(self.dilated_conv3(x))
        x = F.relu(self.dilated_conv4(x))
        x = self.dilated_conv5(x)  # No activation for the output layer
        return x