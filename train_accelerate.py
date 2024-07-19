# Training script for the CustomUNet model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.unet1 import CustomUnet
from losses.losses import get_loss
import argparse
import wandb
from logzero import logger
from dataset import Flare7kpp_Pair_Loader
import yaml
from torcheval.metrics import PeakSignalNoiseRatio
from models.uformer_cmx import Uformer
from dataset import get_loader
from pdb import set_trace as stx
from utils.utils import create_comparision_image
import toml
from accelerate import Accelerator
import os
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Initialize the accelerator
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(project_dir=".",kwargs_handlers=[ddp_kwargs])
os.makedirs("checkpoints", exist_ok=True)

# Set the random seed
set_seed(42)

# Path to your TOML config file
config_file = 'config.toml'

# Load the TOML file
with open(config_file, 'r') as f:
    toml_config = toml.load(f)

# Create sweep configuration
sweep_config = toml_config["sweep_config"]

#Deafult wandb configs
config_defaults = toml_config["wandb_config_defaults"]

# Training Function
def train_fn(model, loss_fn, device,optimizer=None):

    plot_dict = {}
    train_loss_list = []
    val_loss_list =  []
    train_psnr_list = []
    val_psnr_list = []
    val_avg_psnr_list = []
    train_avg_psnr_list = []
    use_wandb = toml_config["use_wandb"]["value"]

    if use_wandb:
        wandb.init(config=config_defaults)
        config = wandb.config

        optim_value = config.optimizer
        lr = config.learning_rate
        num_epochs = config.epochs
        batch_size = config.batch_size

    else:
        config = toml_config   
        optim_value = toml_config["optimizer"]["type"]
        lr = toml_config["optimizer"]["learning_rate"]
        num_epochs = toml_config["training"]["num_epochs"]
        batch_size = toml_config["data"]["batch_size"]
        
    train_loader = get_loader('train', toml_config["data"]["dataset_dir"], batch_size,toml_config["data"]["num_workers"])
    val_loader = get_loader('val', toml_config["data"]["dataset_dir"], batch_size,toml_config["data"]["num_workers"])

    if optim_value=='sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    elif optim_value=='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
    elif optim_value=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    elif optim_value=='nadam':
        optimizer =torch.optim.NAdam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    # Train the model
    model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    accelerator.register_for_checkpointing(model, optimizer, scheduler)

    # Resume training
    checkpoint_dir = "checkpoints"

    # scan directory for checkpoint files
    checkpoint_folders = os.listdir(checkpoint_dir)
    checkpoint_folders_with_path = [os.path.join(checkpoint_dir, folder) for folder in checkpoint_folders]

    # if there are checkpoint files, load the latest one
    if checkpoint_folders_with_path:
        latest_checkpoint = max(checkpoint_folders_with_path, key=os.path.getctime)
        accelerator.load_state(latest_checkpoint)

    # chekpointing path will be like checkpoints/epoch_{epoch}_step_{step}/*
    # get step and epoch from the checkpoint path
        # start_epoch = int(latest_checkpoint.split("/")[1].split("_")[1])
        iters_done = int(latest_checkpoint.split("/")[1].split("_")[-1])
        start_epoch = iters_done // len(train_loader)
        resume_step = iters_done % len(train_loader)
        if accelerator.is_local_main_process:

            logger.info(
                f"Resuming training from epoch {start_epoch}, iter {iters_done} and skipping batches {resume_step}"
            )

    else:
        start_epoch = 0
        resume_step = None
  
    overall_step = 0
    
    psnr = PeakSignalNoiseRatio()

    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss = 0
        if toml_config["training"]["resume"] and epoch == start_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_loader, resume_step)

            overall_step += iters_done

        else:
            active_dataloader = train_loader

        for i, (rgb, depth, flare) in enumerate(active_dataloader):

            optimizer.zero_grad()
            
            # Forward pass
            output = model(flare, depth)

            # Calculate the loss
            loss = loss_fn(output, rgb)
            running_loss += loss.item()

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            psnr.update(rgb, output)
            train_psnr_list.append(psnr.compute().item())

            overall_step += 1

            if overall_step % toml_config["training"]["checkpointing_steps"] == 0:
                save_dir = f"checkpoints/epoch_{epoch}_iters_{overall_step}"
                accelerator.wait_for_everyone()
                accelerator.save_state(save_dir)
                if accelerator.is_local_main_process:
                    logger.info(
                        f"Checkpoint saved at epoch {epoch} and iter {overall_step}"
                    )

        # Print the loss
        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch: {epoch+1}/{num_epochs}, PSNR: {sum(train_psnr_list)/len(train_psnr_list):.4f}, Loss: {running_loss:.4f}"
            )
        train_loss_list.append(running_loss) # For plotting only
        train_avg_psnr_list.append(sum(train_psnr_list)/len(train_psnr_list))

        if use_wandb:
            wandb.log(
                {
                    "train_loss": running_loss,
                    "train_psnr": sum(train_psnr_list) / len(train_psnr_list),
                }
            )

    # TODO: Add validation loop and wandb logging (images) 
    
    return plot_dict, train_avg_psnr_list, val_avg_psnr_list


def main():    
    # Define the device
    device = accelerator.device

    # Get Model
    if toml_config["model"]["type"] == "Uformer":
        model_restoration = Uformer(
            img_size=toml_config["model"]["input_size"],
            embed_dim=toml_config["model"]["embed_dim"],
            depths=toml_config["model"]["depths"],
            win_size=toml_config["model"]["win_size"],
            mlp_ratio=4.0,
            token_projection="linear",
            token_mlp="leff",
            modulator=True,
            shift_flag=False,
        )

    criterion = get_loss(toml_config["model"]["loss"])
    train_fn(model_restoration, criterion, device)

if __name__ == "__main__":
    # Initialize the sweep
    if toml_config["use_wandb"]["value"]:
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep_config, project="Flare", entity="yasharora102")
        wandb.agent(sweep_id, function=main, count=toml_config["use_wandb"]["num_sweeps"])
    else:
        print("No wandb sweep initiated.")
        # print(args)
        main()
