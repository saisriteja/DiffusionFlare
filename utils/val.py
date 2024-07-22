import torch
from tqdm import tqdm
from torcheval.metrics import PeakSignalNoiseRatio
from utils.utils import create_comparision_image
from logzero import logger
import wandb
import os
psnr = PeakSignalNoiseRatio()

def val_script(accelerator,model, loss_fn, plot_dict, val_loss_list, val_psnr_list, val_avg_psnr_list, use_wandb, num_epochs, val_loader, overall_step, epoch, val_out_dir,loss_type):
    
    os.makedirs(val_out_dir, exist_ok=True)
    
    with torch.no_grad():
        model.eval()
        running_val_loss = 0
        plot_list = []
        val_loss = []
        val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
        

        for i, (rgb, depth, flare) in enumerate(val_loader):
            
            model.to(accelerator.device)
            rgb.to(accelerator.device)
            depth.to(accelerator.device)
            flare.to(accelerator.device)

            # Forward pass
            output = model(flare, depth)
            plot_list.append([rgb, depth, flare, output])

            psnr.update(rgb, output)
            val_psnr_list.append(psnr.compute().item())

                    # Calculate the loss
            # loss = loss_fn(output, rgb)
            if loss_type == "Fusionloss_Swin":
                # print("Using Fusion Loss")
                # print(loss_fn)
                loss = loss_fn(flare, depth, output, rgb)
                loss = loss[0]

            else:
                loss = loss_fn(output, rgb)

            running_val_loss += loss.item()
            val_loss.append(loss.item())
            val_pbar.update(1)
        val_pbar.close()

        avg_val_loss = accelerator.gather(torch.tensor(val_loss, device=accelerator.device).unsqueeze(0)).mean().item()
        avg_val_psnr = accelerator.gather(torch.tensor(val_psnr_list, device=accelerator.device).unsqueeze(0)).mean().item()
        
        # Create Val Images
        final_image = create_comparision_image(epoch, plot_list)
        plot_dict[epoch] = plot_list

                # Print the loss
        if accelerator.is_local_main_process:
            logger.info(
                        f"Validation: Epoch: {epoch+1}/{num_epochs}, Iters:{overall_step}, PSNR: {avg_val_psnr:.4f}, Loss: {avg_val_loss:.4f}"
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
            final_image.save(f"{val_out_dir}/epoch_{epoch}_iters_{overall_step}.png")