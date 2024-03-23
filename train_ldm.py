from diffusers import UNet2DModel, DDIMScheduler, VQModel
import torch
import PIL.Image
import numpy as np
import tqdm
import glob
import os
import glob
from torchvision import transforms

from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup



import wandb
wandb.init(project="ddpm-night-imgs")




from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

accelerator = Accelerator()
device = accelerator.device



config = TrainingConfig()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path = '/media/cilab/data/NTIRE/flare/iitm_dataset'):
        self.path = path
        self.files = glob.glob(os.path.join(path, "*.jpg"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = PIL.Image.open(self.files[idx]).convert("RGB")

        tf = transforms.Compose(
                [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        img = tf(img)
        return img


seed = 3

# load all models
unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

# set to cuda
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

unet.to(torch_device)
vqvae.to(torch_device)

#  
dataset = ImageDataset()
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False)


optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler)

vqvae.requires_grad_(False)


import torch.nn.functional as F
from tqdm import tqdm


import cv2



global_step = 0

os.makedirs("images", exist_ok=True)


for epoch in range(config.num_epochs):

    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    unet.train()

    losses = []


    for i, batch in enumerate(tqdm(train_dataloader)):
        latents = vqvae.encode(batch).latents

        # generate the noise
        noise = torch.randn_like(latents)
        bsz = batch.shape[0]

        # sample timesteps
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=torch_device)
        timesteps = timesteps.long()

        with accelerator.accumulate(unet):
            # make the latents noisy
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # predict the noise
            noise_pred = unet(noisy_latents, timesteps).sample

            # loss beign the L2 loss
            loss = F.mse_loss(noise_pred, noise)

            losses.append(loss.item())

            # backprop
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(unet.parameters(), 1.0)

            # if (step + 1) % config.grad_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()



        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1

        # upload all to wandb
        wandb.log(logs, step=global_step)


    if epoch % 10 == 0:
        # accelerator.save(unet.state_dict(), f"unet_{epoch}.pt")
        # accelerator.save(optimizer.state_dict(), f"optimizer_{epoch}.pt")
        # accelerator.save(lr_scheduler.state_dict(), f"lr_scheduler_{epoch}.pt")

        unetp = accelerator.unwrap_model(unet)

        latents = torch.randn(
            (config.eval_batch_size, 3, 32,32),
            generator=torch.Generator().manual_seed(seed),
        )

        # unwrap the unet
        latents = latents.to(accelerator.device)
        latents = latents * scheduler.init_noise_sigma

        # scheduler.num_inference_steps = 500
        scheduler.set_timesteps(10)
        unetp.requires_grad_(False)

        for t in tqdm(scheduler.timesteps):
            latent_model_input = scheduler.scale_model_input(latents, t)

            with torch.no_grad():
                noise_pred = unetp(latent_model_input, t).sample
            latents = scheduler.step(noise_pred,t, latents).prev_sample

        image = vqvae.decode(latents).sample
        image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)
        image = image.permute(0, 2, 3, 1).cpu().numpy()
        # cv2.imwrite(f"sample_{epoch}.png", image[0] * 255.0)
        # stack all the image in a row
        image = np.concatenate(image, axis=1)
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(f"images/samples_{epoch}.png", image)


        # log the image to wandb
        wandb.log({f"samples_{epoch}": [wandb.Image(f"sample_{epoch}.png")]}, step=global_step)
