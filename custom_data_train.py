import wandb
from utils import marginal_prob_std, diffusion_coeff
from tqdm.auto import tqdm
from torchvision.utils import make_grid

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


from diffusers import UNet2DModel

latent_size = 256
model = UNet2DModel(
    sample_size=latent_size,  # the target image resolution
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
# score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))

#from accelerate import Accelerator
#accelerator = Accelerator()

n_epochs = 10
batch_size = 4
lr=1e-4

wandb.init(project="flare_diffusion", entity='adithya-lenka')

from train_data_loader import IITM_Dataset

dataset = IITM_Dataset()
dataloader = DataLoader(dataset, batch_size, shuffle=True)

device = 'cuda:1'
model = model.to(device)

optimizer = Adam(model.parameters(), lr=lr)

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
  loss = torch.mean(torch.sum((score + z)**2, dim=(1,2,3)))
  return loss

for epoch in tqdm(range(n_epochs)):
  avg_loss = 0.
  num_items = 0
  for x in dataloader:
    x = x.to(device)
    loss = loss_fn(model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item()*x.shape[0]
    num_items += x.shape[0]
    wandb.log({"loss":loss})
#  tqdm.set_description('Average Loss: {:5f}'.format(avg_loss/num_items))
  #if (epoch%10)==9:    
  #  torch.save(model.state_dict(), f'ckpts/ckpts{epoch+1}.pth')
  x = next(iter(dataloader)).to(device)
  input_grid = make_grid(x, nrow=2)
  input_grid = wandb.Image(input_grid, caption="Input images")
  wandb.log({"input": input_grid})
  noise_timestep = .5
  t = torch.ones(x.shape[0], device = x.device)*noise_timestep
  z = torch.randn_like(x).to(device)
  std = marginal_prob_std_fn(t).to(device)
  noisy_x = x + z*std[:,None, None, None]
  batch_size = x.shape[0]
  eps=1e-3
  time_steps = torch.linspace(noise_timestep, eps, 200, device=device)
  step_size = time_steps[0] - time_steps[1]

  with torch.no_grad():
    for time_step in tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff_fn(batch_time_step)
      mean_x = noisy_x + (g**2)[:, None, None, None]*model(noisy_x, batch_time_step).sample*step_size
      noisy_x = mean_x + torch.sqrt(step_size)*g[:, None, None, None]*torch.randn_like(noisy_x)

  mean_x = mean_x.clamp(0.0, 1.0)
  sample_grid = make_grid(mean_x, nrow=2)
  sample_grid = wandb.Image(sample_grid, caption="Input images")
  wandb.log({"sample": sample_grid})


# print(model)

#print("Output shape:", model(x, timestep=0).sample.shape)




# loss, score = loss_fn(model, x, marginal_prob_std=marginal_prob_std_fn)

# import wandb
# wandb.init(project="flarediffusion", entity="saisritejakuppa")



# print(score.shape)
# images = wandb.Image(score, caption="Top: Output, Bottom: Input")

# x = wandb.Image(x, caption="Top: Output, Bottom: Input")

# wandb.log({"input": x})
# wandb.log({"output": images})
# wandb.log({"loss": loss})
# print(loss)