import torch
import torch.nn as nn
from torchvision.models import vgg19
import torchvision.models.vgg as vgg
import torch.nn.functional as F
from .loss_ssim import ssim

def get_loss(loss):
    if loss == "L1":
        return L1_loss
    else:
        return NotImplementedError

def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

# Used by Flare7kpp (0.5 weighting)
class L_Abs_pure(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_Abs_pure, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,x,flare_gt):
        flare_loss=torch.abs(x-flare_gt)
        Abs_loss=torch.mean(flare_loss)
        return self.loss_weight*Abs_loss
    
# Used by Flare7kpp (0.5 weighting)
class L_percepture(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_percepture, self).__init__()
        self.loss_weight=loss_weight
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        model=model.cuda()
        model = model.eval()
        # Freeze VGG19 #
        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        self.mae_loss = nn.L1Loss()
        self.selected_feature_index=[2,7,12,21,30]
        self.layer_weight=[1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
    
    def extract_feature(self,x):
        selected_features = []
        for i,model in enumerate(self.vgg):
            x = model(x)
            if i in self.selected_feature_index:
                selected_features.append(x.clone())
        return selected_features

    def forward(self, source, target):
        source_feature = self.extract_feature(source)
        target_feature = self.extract_feature(target)
        len_feature=len(source_feature)
        perceptual_loss=0
        for i in range(len_feature):
            perceptual_loss+=self.mae_loss(source_feature[i],target_feature[i])*self.layer_weight[i]
        return self.loss_weight*perceptual_loss
    

# used by FusionMamba (other is L1)
class ERGAS(torch.nn.Module):
    def __init__(self, ratio=4):
        super().__init__()
        self.ratio = ratio

    def forward(self, img, gt):
        b, c, _, _ = img.shape
        a1 = torch.mean((img - gt) ** 2, dim=(-2, -1))
        a2 = torch.mean(gt, dim=(-2, -1)) ** 2
        com = (a1 / a2).view(b, c)
        summ = torch.sum(com, dim=-1)
        ergas = 100 * (1 / self.ratio) * ((summ / c) ** 0.5)
        ergas = ergas.mean()
        return ergas
    

# From CDDFuse [https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/utils/loss.py]    
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()




# SwinFusion [https://github.com/Linfeng-Tang/SwinFusion/blob/master/models/loss_vif.py]

# Texture loss
class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient
        
# SSIM Loss
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

# intensity Loss
class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):        
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

# Combined Loss
class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM
    

