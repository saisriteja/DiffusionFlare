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

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Path to your TOML config file
config_file = 'config.toml'

# Load the TOML file
with open(config_file, 'r') as f:
    toml_config = toml.load(f)

print(toml_config)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="DiffusionFlare/sample_dataset/")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=1)
# parser.add_argument("--use_wandb", type=bool, default=False)
parser.add_argument("--use_wandb", type=bool, default=True)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_sweeps", type=int, default=3)
args = parser.parse_args()


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
    model.train()
    model.to(device)

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
        logger.info(
            f"Epoch: {epoch+1}/{num_epochs}, PSNR: {sum(train_psnr_list)/len(train_psnr_list):.4f}, Loss: {running_loss:.4f}"
        )
        train_loss_list.append(running_loss)
        train_avg_psnr_list.append(sum(train_psnr_list)/len(train_psnr_list))

        if use_wandb:
            wandb.log(
                {
                    "train_loss": running_loss,
                    "train_psnr": sum(train_psnr_list) / len(train_psnr_list),
                }
            )

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

            # Create Val Images
            final_image = create_comparision_image(epoch, plot_list)
            plot_dict[epoch] = plot_list

            # Print the loss
            logger.info(
                f"Validation: Epoch: {epoch+1}/{num_epochs}, PSNR: {sum(val_psnr_list)/len(val_psnr_list):.4f}, Loss: {running_val_loss:.4f}"
            )
            val_loss_list.append(running_val_loss)
            val_avg_psnr_list.append(sum(val_psnr_list)/len(val_psnr_list))

            if use_wandb:
                wandb.log(
                    {
                        "val_loss": running_val_loss,
                        "val_psnr": sum(val_psnr_list) / len(val_psnr_list),
                        "output_image": wandb.Image(final_image),
                    }
                )
            else:
                final_image.save("val_out.png")

    # # TODO: remove this
    # if use_wandb:
    #     wandb.save(f"epoch_{epoch+1}.pth")

    return plot_dict, train_avg_psnr_list, val_avg_psnr_list


def main():    
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    print(args)
    # Initialize the sweep
    if toml_config["use_wandb"]["value"]:
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep_config, project="Flare", entity="yasharora102")
        wandb.agent(sweep_id, function=main, count=toml_config["use_wandb"]["num_sweeps"])
    else:
        print("No wandb sweep initiated.")
        print(args)
        main()
