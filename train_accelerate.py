# Training script for the CustomUNet model
import torch
import torch.nn as nn
from models.unet1 import CustomUnet
from losses.losses import get_loss
import wandb
from logzero import logger
from dataset import Flare7kpp_Pair_Loader
import yaml
from tqdm import tqdm
from models.uformer_cmx import Uformer
from dataset import get_loader
from pdb import set_trace as stx
import toml
from accelerate import Accelerator
import os
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from models.swin_fusion.swin_fusion_model import SwinFusion
from utils.val import val_script
from losses.loss_gt import fusion_loss_gt
from schedulers.schedulers import get_scheduler
from datetime import datetime
import logzero
import matplotlib.pyplot as plt
from models.Emma.Ufuser import Ufuser
from models.Emma.Unet5 import UNet5 as unet
from utils.emma_utils import Transformer
# from models.MambaDFuse.mambadfuse import MambaDFuse

# Path to your TOML config file
config_file = 'config.toml'

# Load the TOML file
with open(config_file, 'r') as f:
    toml_config = toml.load(f)

# Initialize the accelerator
if toml_config["model"]["type"] == "EMMA":
    accelerator = Accelerator(project_dir=".")

else:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=".",kwargs_handlers=[ddp_kwargs])

os.makedirs("checkpoints", exist_ok=True)

# Set the logger
logger = logzero.setup_default_logger(disableStderrLogger=True)
# get current time and date
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
logzero.logfile(f"logs/{current_time}.log")

# Set the random seed
set_seed(42)

# Create sweep configuration
sweep_config = toml_config["sweep_config"]

#Deafult wandb configs
config_defaults = toml_config["wandb_config_defaults"]

# Training Function
def train_fn(model_list, loss_fn, device,optimizer=None,tran=None):

    plot_dict = {}
    train_loss_list = []
    val_loss_list =  []
    train_psnr_list = []
    val_psnr_list = []
    val_avg_psnr_list = []
    train_avg_psnr_list = []
    use_wandb = toml_config["use_wandb"]["value"]
    if toml_config["model"]["type"] == "EMMA":
        model = model_list[2]
    else:
        model = model_list
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
    # train_loader = get_loader('small_train', toml_config["data"]["dataset_dir"], batch_size,toml_config["data"]["num_workers"])
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

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    scheduler = get_scheduler(optimizer, toml_config["scheduler"])
    # print(scheduler)
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
                f"Resuming training from epoch {start_epoch}, iter {iters_done} and skipping batches {resume_step}, lr: {scheduler.get_last_lr()}"
            )

    else:
        start_epoch = 0
        resume_step = None
  
    overall_step = 0
    
    # psnr = PeakSignalNoiseRatio()
    train_loss_list_plot = []
    
    for epoch in range(start_epoch,num_epochs):
        # if toml_config["model"]["type"] == "EMMA":
        #     model[2].train()
        # else:
        #     model.train()

        if toml_config["training"]["resume"] and epoch == start_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_loader, resume_step)
            overall_step += iters_done

        else:
            active_dataloader = train_loader
        
        running_loss = 0
        for i, (rgb, depth, flare) in tqdm(enumerate(active_dataloader), total=len(active_dataloader)):

            optimizer.zero_grad()
            
            # Forward pass
            if toml_config["model"]["type"] == "EMMA":
                output = model(flare, depth)
                F2Vmodel = model_list[0].to(accelerator.device)
                F2Imodel = model_list[1].to(accelerator.device)
                Ft = tran.apply(output)
                Ft_caret= model(F2Imodel(Ft),F2Vmodel(Ft))
                loss=loss_fn(F2Vmodel(output),flare)+loss_fn(F2Imodel(output),depth)+0.1*loss_fn(Ft,Ft_caret)

            else:
                output = model(flare, depth)

            # Calculate the loss
            if toml_config["model"]["loss"] == "Fusionloss_Swin":
                # print("Using Fusion Loss")
                # print(loss_fn)
                loss = loss_fn(flare, depth, output, rgb)
                loss = loss[0]
                running_loss += loss
            else:
                loss = loss_fn(output, rgb)
                running_loss += loss.item()

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            overall_step += 1
            train_loss_list_plot.append(loss.item())

            if overall_step % toml_config["data"]["val_freq"] == 0:
                if accelerator.is_local_main_process:
                    logger.info(f"Validation started with {overall_step} iterations")
                val_script(accelerator,model, loss_fn, plot_dict, val_loss_list, val_psnr_list, val_avg_psnr_list, use_wandb, num_epochs, val_loader, overall_step, epoch, toml_config["data"]["val_out_dir"],toml_config["model"]["loss"])
                model.train()
                accelerator.wait_for_everyone()

            if overall_step % toml_config["training"]["checkpointing_steps"] == 0:
                save_dir = f"checkpoints/epoch_{epoch}_iters_{overall_step}"
                accelerator.wait_for_everyone()
                accelerator.save_state(save_dir)
                if accelerator.is_local_main_process:
                    logger.info(
                        f"Checkpoint saved at epoch {epoch} and iter {overall_step}"
                    )

            plot_steps = toml_config["training"]["plot_steps"]
            if overall_step % plot_steps == 0:
                plot_loss(train_loss_list_plot)

        # Print the loss
        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch: {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, lr: {scheduler.get_last_lr()}"
            )
        train_loss_list.append(running_loss) # For plotting onlys
        # train_avg_psnr_list.append(sum(train_psnr_list)/len(train_psnr_list))

        if use_wandb:
            wandb.log(
                {
                    "train_loss": running_loss,
                    # "train_psnr": sum(train_psnr_list) / len(train_psnr_list),
                }
            )        
    return plot_dict

def plot_loss(train_loss_list):
    plt.plot(train_loss_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('train_loss.png')

def main():    
    # Define the device
    device = accelerator.device
    tran = None # For EMMA
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
    
    elif toml_config["model"]["type"] == "SwinFusion":
        upscale = toml_config["model"]["swin_upscale"]
        window_size = toml_config["model"]["win_size"]
        height = (1024 // upscale // window_size + 1) * window_size
        width = (720 // upscale // window_size + 1) * window_size
        model_restoration = SwinFusion(upscale=upscale, 
                        img_size=(height, width),
                        # patch_size= toml_config["model"]["swin_patch_size"],
                        in_chans = 3,
                            window_size=window_size, img_range=1., 
                            depths=toml_config["model"]["swin_depths"],
                            embed_dim=toml_config["model"]["swin_embed_dim"], 
                            num_heads=toml_config["model"]["swin_num_heads"], 
                            mlp_ratio=2, 
                            upsampler='pixelshuffledirect')
        print(height, width, model_restoration.flops() / 1e9)

    elif toml_config["model"]["type"] == "EMMA":

        F2V_path=r'models/Emma/Av.pth'
        F2I_path=r'models/Emma/Ai.pth'

        F2Vmodel = unet() 
        F2Vmodel.load_state_dict(torch.load(F2V_path))
        F2Vmodel.eval()

        F2Imodel = unet()
        F2Imodel.load_state_dict(torch.load(F2I_path))
        F2Imodel.eval()

        shift_num=3
        rotate_num=3
        flip_num=3

        model=Ufuser()
        model_restoration = [F2Vmodel,F2Imodel,model]
        tran = Transformer(shift_num, rotate_num, flip_num)
        
    criterion = get_loss(toml_config["model"]["loss"])
    train_fn(model_restoration, criterion, device,tran = tran)

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
