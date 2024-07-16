import torch
import torch.nn as nn
from torchvision.models import vgg19
import torchvision.models.vgg as vgg

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