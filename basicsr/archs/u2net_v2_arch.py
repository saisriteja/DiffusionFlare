import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from .fusion_mamba import FusionMamba
from basicsr.utils.registry import ARCH_REGISTRY
from pdb import set_trace as stx

class ResBlock2D(nn.Module):
    def __init__(self, dim, res_se_ratio):
        super().__init__()
        hidden_dim = int(res_se_ratio * dim)
        self.conv0 = nn.Conv2d(dim, hidden_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_dim, dim, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class PixelShuffle(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(dim, dim*(scale**2), 3, 1, 1, bias=False),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale, upsample='default'):
        super().__init__()
        if upsample == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'bicubic':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'pixelshuffle':
            self.up = nn.Sequential(
                PixelShuffle(in_channels, scale),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = x1 + x2
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale, downsample='default'):
        super().__init__()
        if downsample == 'maxpooling':
            self.down = nn.Sequential(
                nn.MaxPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.down(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, scale=2, sample_mode='down'):
        super().__init__()
        self.fm = FusionMamba(in_channels, H, W)
        if sample_mode == 'down':
            self.sample = Down(in_channels, out_channels, scale)
        elif sample_mode == 'up':
            self.sample = Up(in_channels, out_channels, scale)

    def forward(self, pan, ms, pan_pre=None, ms_pre=None):
        pan, ms = self.fm(pan, ms)
        if pan_pre is None:
            pan_skip = pan
            ms_skip = ms
            pan = self.sample(pan)
            ms = self.sample(ms)
            return pan, ms, pan_skip, ms_skip
        else:
            pan = self.sample(pan, pan_pre)
            ms = self.sample(ms, ms_pre)
            return pan, ms


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim,hidden_dim,1,1,0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim ,3,1,1)

        self.act =nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1,x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:,:self.p_dim,:,:] = self.act(self.conv_1(x[:,:self.p_dim,:,:]))
            x = self.conv_2(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)

        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        _,_,h,w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.smfa = SMFA(dim)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x


@ARCH_REGISTRY.register()
class U2Net_v2(nn.Module):
    def __init__(self, dim, pan_dim, ms_dim, H=64, W=64, scale=4, use_6_channel = False):
        super().__init__()
        self.use_6_channel = use_6_channel
        self.upsample = PixelShuffle(ms_dim, scale)
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(pan_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(ms_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )
        # output to 6 channels
        if self.use_6_channel:
            self.to_6_channels = nn.Sequential(
                nn.Conv2d(ms_dim, dim, 3, 1, 1),
                nn.LeakyReLU(),
                nn.Conv2d(dim, 6, 3, 1, 1),
                nn.Sigmoid()
            )
        # dimension for each stage
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        # main body
        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down')
        self.fmb0 = FMB(dim1)
        self.stage1 = Stage(dim1, dim2, H//2, W//2, sample_mode='down')
        self.fmb1 = FMB(dim2)
        self.stage2 = Stage(dim2, dim1, H//4, W//4, sample_mode='up')
        self.fmb2 = FMB(dim1)
        self.stage3 = Stage(dim1, dim0, H//2, W//2, sample_mode='up')
        self.fmb3 = FMB(dim0)
        self.stage4 = FusionMamba(dim0, H, W)
        self.fmb4 = FMB(dim0)

    def forward(self, ms, pan):
        ms = self.upsample(ms)
        skip = ms
        pan = self.raise_pan_dim(pan)
        ms = self.raise_ms_dim(ms)

        # main body
        pan, ms, pan_skip0, ms_skip0 = self.stage0(pan, ms)
        ms = self.fmb0(ms)
        pan, ms, pan_skip1, ms_skip1 = self.stage1(pan, ms)
        ms = self.fmb1(ms)
        pan, ms = self.stage2(pan, ms, pan_skip1, ms_skip1)
        ms = self.fmb2(ms)
        pan, ms = self.stage3(pan, ms, pan_skip0, ms_skip0)
        ms = self.fmb3(ms)
        _, ms = self.stage4(pan, ms)
        ms = self.fmb4(ms)

        output = self.to_hrms(ms) + skip
        
        # 6 Channel Output for flare7kpp (set use_6_channel=True)
        if self.use_6_channel:
            output = self.to_6_channels(output)
        
        return output
    

if __name__ == '__main__':
    model = U2Net_v2(32, 1, 3, 256,256, 1, True)
    rgb = torch.randn(1,3,256,256)
    depth = torch.randn(1,1,256,256)
    model.to('cuda') 
    rgb, depth = rgb.to('cuda'), depth.to('cuda')
    out = model(rgb,depth)
    # stx()


    # # from fvcore.nn import FlopCountAnalysis
    # # flop_counter = FlopCountAnalysis(model, (rgb, depth))
    # # print(flop_counter.total()/1e9)


    # # Params
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 'M')


    # from thop import profile

    # macs, params = profile(model, inputs=(rgb, depth))
    # print(macs/1e9, 'G MACs')
    # print(params/1e6, 'M')


    summary(model, input_size=[(1,3, 256, 256), (1,1, 256, 256)], device='cuda')