import wandb
from utils import marginal_prob_std, diffusion_coeff



#@title Set up the SDE
import functools

device = 'cuda:1' #@param ['cuda', 'cpu'] {'type':'string'}

sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)


#@title Training (double click to expand or collapse)
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm.notebook


from utils import ScoreNet, loss_fn


# score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))

from accelerate import Accelerator
accelerator = Accelerator()





img_size = 256








class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        # get me a 512,512 image
        img = torch.rand(3, img_size, img_size)
        
        return img
    

train_dataloader = DataLoader(Dataset(), batch_size=4, shuffle=True, num_workers=4)

for x in train_dataloader:
    x = x.to(device)
    print(x.shape)
    break





from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=img_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)



# print(model)
model = model.to('cuda:1')
print("Output shape:", model(x, timestep=0).sample.shape)




def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t).sample
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss, score



loss, score = loss_fn(model, x, marginal_prob_std=marginal_prob_std_fn)

import wandb
wandb.init(project="flarediffusion", entity="saisritejakuppa")



print(score.shape)
images = wandb.Image(score, caption="Top: Output, Bottom: Input")

x = wandb.Image(x, caption="Top: Output, Bottom: Input")

wandb.log({"input": x})
wandb.log({"output": images})
wandb.log({"loss": loss})
print(loss)