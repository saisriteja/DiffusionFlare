# Training script for the CustomUNet model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.unet1 import CustomUnet
from losses.losses import L1_loss
import argparse
import wandb
from logzero import logger
from dataset import Flare7kpp_Pair_Loader
import yaml
from torcheval.metrics import PeakSignalNoiseRatio
from models.uformer_cmx import Uformer
from dataset import get_loader
from pdb import set_trace as stx


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="DiffusionFlare/sample_dataset/")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--use_wandb", type=bool, default=True)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--learning_rate", type=float, default=0.001)
args = parser.parse_args()


# Create sweep configuration
sweep_config = {
    "method": "random",  # grid, random
    "metric": {"name": "val_psnr", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [2, 5, 10]},
        "batch_size": {"values": [4, 6, 8, 12]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]},
        "optimizer": {"values": ["adam", "sgd", "rmsprop"]},
    },
}

config_defaults = {
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
}

# Training Function
def train_fn(model, loss_fn, device,optimizer=None):
    
    plot_dict = {}
    train_loss_list = []
    val_loss_list =  []
    train_psnr_list = []
    val_psnr_list = []
    val_avg_psnr_list = []
    train_avg_psnr_list = []

    wandb.init(config=config_defaults)
    config = wandb.config
    train_loader = get_loader('train', 'sample_dataset/Flickr24K', config.batch_size)
    val_loader = get_loader('val', 'sample_dataset/Flickr24K', config.batch_size)

    if config.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=config.learning_rate)
    elif config.optimizer=='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),lr=config.learning_rate)
    elif config.optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    elif config.optimizer=='nadam':
        optimizer =torch.optim.NAdam(model.parameters(),lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    # Train the model
    model.train()
    model.to(device)
    num_epochs = config.epochs

    for epoch in range(num_epochs):
        running_loss = 0 
        for i, (rgb, depth, flare) in enumerate(train_loader):
            
            # Move the data to GPU
            rgb = rgb.to(device)
            depth = depth.to(device)
            flare = flare.to(device)

            # Forward pass
            output = model(flare, depth)

            # Calculate the loss
            loss = loss_fn(output, rgb)
            running_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = PeakSignalNoiseRatio()
            psnr.update(rgb, output)
            train_psnr_list.append(psnr.compute().item())

        # Print the loss
        logger.info(f"Epoch: {epoch+1}/{num_epochs}, PSNR: {sum(train_psnr_list)/len(train_psnr_list):.4f}, Loss: {running_loss:.4f}")
        train_loss_list.append(running_loss)
        train_avg_psnr_list.append(sum(train_psnr_list)/len(train_psnr_list))
        
        wandb.log({"train_loss": running_loss, "train_psnr": sum(train_psnr_list)/len(train_psnr_list)})
    # Evaluate the model on the validation set
        with torch.no_grad():
            running_val_loss = 0
            plot_list = []
            for i, (rgb, depth, flare) in enumerate(val_loader):

                # Move the data to GPU
                rgb = rgb.to(device)
                depth = depth.to(device)
                flare = flare.to(device)

                # Forward pass
                output = model(flare, depth)
                
                plot_list.append([rgb, depth, flare, output])

                psnr = PeakSignalNoiseRatio()
                psnr.update(rgb, output)
                val_psnr_list.append(psnr.compute().item())

                # Calculate the loss
                loss = loss_fn(output, rgb)
                running_val_loss += loss.item()

            plot_dict[epoch] = plot_list

            # Print the loss
            logger.info(f"Validation: Epoch: {epoch+1}/{num_epochs}, PSNR: {sum(val_psnr_list)/len(val_psnr_list):.4f}, Loss: {running_val_loss:.4f}")
            val_loss_list.append(running_val_loss)
            val_avg_psnr_list.append(sum(val_psnr_list)/len(val_psnr_list))

            wandb.log({"val_loss": running_val_loss, "val_psnr": sum(val_psnr_list)/len(val_psnr_list)})
    
    wandb.save(f"epoch_{epoch+1}.pth")
    return plot_dict, train_avg_psnr_list, val_avg_psnr_list


def main():    
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    input_size = 256
    arch = Uformer
    # depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    depths=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    model_restoration = Uformer(img_size=input_size, embed_dim=16,depths=depths,
                    win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
    
    criterion = L1_loss
    train_fn(model_restoration, criterion, device)

    # TODO:  Create and save plots


if __name__ == "__main__":
    
    print(args)
    # stx()
    # Initialize the sweep
    if args.use_wandb:
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep_config, project="Flare", entity="yasharora102")
        wandb.agent(sweep_id, function=main, count=3)
    else:
        print("No wandb sweep initiated.")
        print(args)
        main()
