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


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()


# Create sweep configuration
sweep_config = {
    "method": "random",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128]},
        "lr": {"values": [0.001, 0.01, 0.1]},
    },
}

# Training Function
def train_fn(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    # Dataloader output:
    # {'gt': adjust_gamma_reverse(base_img),'flare': adjust_gamma_reverse(flare_img),'lq': adjust_gamma_reverse(merge_img),'mask': flare_mask,'gamma': gamma}
    
    for i, data in enumerate(dataloader):
        images, labels = data['lq'], data['gt']
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def main():
    # Load the dataset, haven't defined any transforms as of yet
    opts = yaml.load(open("options/config.yaml", "r"), Loader=yaml.FullLoader)

    dataloader = Flare7kpp_Pair_Loader(opts)
    # Initialize the model
    model = CustomUnet(3, 3).to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = L1_loss
    
    for epoch in range(args.epochs):
        loss = train_fn(model, dataloader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        wandb.log({"Loss": loss, "Epoch": epoch})



if __name__ == "__main__":
    # Initialize the sweep
    wandb.init(project="custom-unet")
    sweep_id = wandb.sweep(sweep=sweep_config, project="my-first-sweep")
    wandb.agent(sweep_id, function=main, count=1)

