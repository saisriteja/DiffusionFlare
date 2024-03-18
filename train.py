import wandb
from utils import marginal_prob_std, diffusion_coeff



#@title Set up the SDE
import functools

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

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






n_epochs =   5#@param {'type':'integer'}
## size of a mini-batch
batch_size =  32 #@param {'type':'integer'}
## learning rate
lr=1e-4 #@param {'type':'number'}

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
optimizer = Adam(score_model.parameters(), lr=lr)


score_model, optimizer, data_loader = accelerator.prepare(score_model, optimizer, data_loader)



import wandb

wandb.init(project="flarediffusion", entity="saisritejakuppa")


# tqdm_epoch = tqdm.notebook.trange(n_epochs)
for epoch in range(n_epochs):
  avg_loss = 0.
  num_items = 0

  print("epochs",epoch)
  for x, y in data_loader:
    loss, score = loss_fn(score_model, x, marginal_prob_std_fn)

    images = wandb.Image(score, caption="Top: Output, Bottom: Input")
    wandb.log({"examples": images})
    wandb.log({"loss": loss})

    quit()

    optimizer.zero_grad()
    # loss.backward() 
    accelerator.backward(loss)   
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
#   tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
  if (epoch%10)==9:
    torch.save(score_model.state_dict(), f'ckpts/ckpt{epoch+1}.pth')