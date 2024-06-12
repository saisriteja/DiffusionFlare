import torch
import torch.nn as nn
import numpy as np
from pdb import set_trace as stx
from unet1 import CustomUnet, EncoderBlock, DecoderBlock
from net_utils import FeatureRectifyModule, FeatureFusionModule


class UnetCMX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetCMX, self).__init__()

        # Layer for Image; will be genearlly a Transformer Block
        self.layer_level_1 = EncoderBlock(in_channels, out_channels)

        # Layer for Depth Image; will be genearlly a Transformer Block
        self.layer_depth_img_level_1 = EncoderBlock(in_channels, out_channels)

        self.FRM = FeatureRectifyModule(dim=32, reduction=1)

        self.FFM = FeatureFusionModule(dim=32, reduction=1, num_heads=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.decoder_level_1 = DecoderBlock(64, 64)

    def forward(self, x, xd):
        pass
        # enc_out = self.layer_level_1(x)
        # # xd = self.layer_depth_img_level_1(xd)
        # # x, xd = self.FRM(x, xd)

        # out = self.pool(enc_out)
        # # xd = self.pool(xd)

        # out = self.upconv(out)
        # # xd = self.upconv(xd)
        # # out = self.FFM(x, xd)

        # out = torch.cat([enc_out, out], dim=1)
        # stx()
        # out = self.decoder_level_1(out)
        # return out


if __name__ == "__main__":
    model = UnetCMX(3, 3)
    x = torch.randn(1, 3, 256, 256)
    xd = torch.randn(1, 3, 256, 256)
    out = model(x, xd)
    print(out.shape)
