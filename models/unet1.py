import torch
import torch.nn as nn
import numpy as np
from pdb import set_trace as stx

class CustomUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomUnet, self).__init__()

        # 256x256x3 -> 256x256x32
        self.encoder_level_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 256x256x32 -> 128x128x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 128x128x32 -> 256x256x32
        self.upconv = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        
        # 256x256x64 -> 256x256x3
        self.decoder_level_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        enc_level_1 = self.encoder_level_1(x)
        out = self.pool(enc_level_1)
        out = self.upconv(out)
        out = torch.cat([enc_level_1, out], dim=0)
        out = self.decoder_level_1(out)
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
    

if __name__ == '__main__':
    model = CustomUnet(3, 3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
