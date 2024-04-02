import toml
from dataloadingutils.dataloading_ops import Flare7kpp_Pair_Loader
from logzero import logger
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch
from utils.discriminator import define_D
from models.generator import define_G
from utils import losses
import os
import wandb


from utils.replay_pool import ReplayPool
replay_pool = ReplayPool(10)

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

def make_generator(name = 'pix2pix'):
    if name == 'pix2pix':
        logger.debug("loading pix2pix model")
        gen = define_G(input_nc = 3, output_nc = 6,
        ngf = 64, netG = "global", norm = "instance",
        n_downsample_global = 3, n_blocks_global = 9, 
        n_local_enhancers = 1, n_blocks_local = 3)

    elif name == 'uformer':
        pass


    else:
        logger.error("model not found")
    return gen

from utils.moving_average import moving_average
from tqdm import tqdm

def process_loss(log, losses, weights = None):
    loss = 0
    for k in losses:
        if k not in log:
            log[k] = 0
        log[k] += losses[k].item() * (weights[k] if weights is not None else 1)
        loss = loss + losses[k]

        # log the loss
        wandb.log({k + '_og': losses[k].item()})
        wandb.log({k: losses[k].item() * (weights[k] if weights is not None else 1)})
    return loss



class Losses:
    def __init__(self):
        criterionGAN = losses.GANLoss(use_lsgan=True)
        criterionFeat = torch.nn.L1Loss()
        criterionVGG = losses.VGGLoss()

        criterionFeat, criterionGAN, criterionVGG = accelerator.prepare(criterionFeat, criterionGAN, criterionVGG)

        self.criterionGAN = criterionGAN
        self.criterionFeat = criterionFeat
        self.criterionVGG = criterionVGG

    def calc_G_losses(self,data, generator, discriminator):

        lq = data['lq']
        gt = data['gt']
        flare = data['flare']
        gamma = data['gamma']

        output_gen  =  generator(lq)
        deflare,flare_hat,merge_hat = predict_flare_from_6_channel(input_tensor=output_gen,gamma=gamma)

        target = dict()
        target['deflare'] = deflare
        target['flare_hat'] = flare_hat
        target['merge'] = merge_hat

        l1_flare = self.criterionFeat(target['flare_hat'], data['flare'])
        l1_base  = self.criterionFeat(target['deflare'], data['gt'])

        loss_vgg = self.criterionVGG(deflare, gt)
        pred_fake = discriminator(torch.cat([lq, deflare], axis=1))
        loss_adv = self.criterionGAN(pred_fake, 1)

        with torch.no_grad():
            pred_true = discriminator(torch.cat([lq, gt], axis=1))

        loss_adv_feat = 0
        adv_feats_count = 0        
        for d_fake_out, d_true_out in zip(pred_fake, pred_true):
            for l_fake, l_true in zip(d_fake_out[: -1], d_true_out[: -1]):
                loss_adv_feat = loss_adv_feat + self.criterionFeat(l_fake, l_true)
                adv_feats_count += 1
        loss_adv_feat = 1*(4/adv_feats_count)*loss_adv_feat

        return {"G_vgg": loss_vgg, "G_adv": loss_adv, 
                "G_adv_feat": loss_adv_feat, 'G_l1_flare': l1_flare, 
                'G_l1_base': l1_base}

    def calc_D_losses(self,data, generator, discriminator, replay_pool):

        lq = data['lq']
        gt = data['gt']
        flare = data['flare']
        gamma = data['gamma']

        with torch.no_grad():
            output_gen  =  generator(lq)
            # fake = replay_pool.query({"input": data.detach(), "output": output_gen.detach()})
            # fake = replay_pool.query({"input": data, "output": output_gen})
            deflare,flare_hat,merge_hat = predict_flare_from_6_channel(input_tensor=output_gen,gamma=gamma)

        pred_true = discriminator(torch.cat([lq, gt], axis=1))
        loss_true = self.criterionGAN(pred_true, 1)
        pred_fake = discriminator(torch.cat([lq, deflare], axis=1))
        loss_false = self.criterionGAN(pred_fake, 0)

        wandb.log({"D_true": loss_true.item(), "D_false": loss_false.item()})

        return {"D_true": loss_true, "D_false": loss_false}




def adjust_gamma_reverse(image: torch.Tensor, gamma):
    #gamma=torch.Tensor([gamma]).cuda()
    gamma=1/gamma.float().cuda()
    gamma_tensor=torch.ones_like(image)
    gamma_tensor=gamma.view(-1,1,1,1)*gamma_tensor
    image=torch.pow(image,gamma_tensor)
    out= torch.clamp(image, 0.0, 1.0)
    return out

def adjust_gamma(image: torch.Tensor, gamma):
    #image is in shape of [B,C,H,W] and gamma is in shape [B]
    gamma=gamma.float().cuda()
    gamma_tensor=torch.ones_like(image)
    gamma_tensor=gamma.view(-1,1,1,1)*gamma_tensor
    image=torch.pow(image,gamma_tensor)
    out= torch.clamp(image, 0.0, 1.0)
    return out



def predict_flare_from_6_channel(input_tensor,gamma):
    #the input is a tensor in [B,C,H,W], the C here is 6

    deflare_img=input_tensor[:,:3,:,:]
    flare_img_predicted=input_tensor[:,3:,:,:]

    merge_img_predicted_linear=adjust_gamma(deflare_img,gamma)+adjust_gamma(flare_img_predicted,gamma)
    merge_img_predicted=adjust_gamma_reverse(torch.clamp(merge_img_predicted_linear, 1e-7, 1.0),gamma)
    return deflare_img,flare_img_predicted,merge_img_predicted






def test(epoch, iteration, generator_ema, val_loader):
    
    with torch.no_grad():
        generator_ema.eval()
        for data in val_loader:
            lq = data['lq']
            gt = data['gt']
            flare = data['flare']
            gamma = data['gamma']
            output_gen  =  generator_ema(lq)
            deflare,flare_hat,merge_hat = predict_flare_from_6_channel(input_tensor=output_gen,gamma=gamma)
            break
        generator_ema.train()


        # save the image in stack
        for i in range(4):
            img = np.concatenate([lq[i].cpu().numpy().transpose(1,2,0), deflare[i].cpu().numpy().transpose(1,2,0), gt[i].cpu().numpy().transpose(1,2,0), flare[i].cpu().numpy().transpose(1,2,0), flare_hat[i].cpu().numpy().transpose(1,2,0), merge_hat[i].cpu().numpy().transpose(1,2,0)], axis=1)
            cv2.imwrite(f"output/images/{epoch}_{iteration}_{i}.png", img*255)
            cv2.imwrite(f"output/images/{epoch}_{iteration}_{i}_lq.png", lq[i].cpu().numpy().transpose(1,2,0)*255)
            cv2.imwrite(f"output/images/{epoch}_{iteration}_{i}_gt.png", gt[i].cpu().numpy().transpose(1,2,0)*255)
            cv2.imwrite(f"output/images/{epoch}_{iteration}_{i}_flare.png", flare[i].cpu().numpy().transpose(1,2,0)*255)
            cv2.imwrite(f"output/images/{epoch}_{iteration}_{i}_deflare.png", deflare[i].cpu().numpy().transpose(1,2,0)*255)
            cv2.imwrite(f"output/images/{epoch}_{iteration}_{i}_flare_hat.png", flare_hat[i].cpu().numpy().transpose(1,2,0)*255)
            cv2.imwrite(f"output/images/{epoch}_{iteration}_{i}_merge_hat.png", merge_hat[i].cpu().numpy().transpose(1,2,0)*255)

        # stack all the images
        img = np.concatenate([lq[0].cpu().numpy().transpose(1,2,0), 
                              deflare[0].cpu().numpy().transpose(1,2,0), 
                              gt[0].cpu().numpy().transpose(1,2,0), 
                              flare[0].cpu().numpy().transpose(1,2,0), 
                              flare_hat[0].cpu().numpy().transpose(1,2,0), 
                              merge_hat[0].cpu().numpy().transpose(1,2,0)], axis=1)
        

        for i in range(1, 4):
            img = np.concatenate([img, np.concatenate([lq[i].cpu().numpy().transpose(1,2,0), 
                                                       deflare[i].cpu().numpy().transpose(1,2,0),
                                                        gt[i].cpu().numpy().transpose(1,2,0), 
                                                       flare[i].cpu().numpy().transpose(1,2,0), 
                                                       flare_hat[i].cpu().numpy().transpose(1,2,0), 
                                                       merge_hat[i].cpu().numpy().transpose(1,2,0)], axis=1)], axis=0)
        
        
        cv2.imwrite(f"output/images/{epoch}_{iteration}.png", img*255)

        # log the image
        wandb.log({"image": [wandb.Image(img*255, caption=f"epoch_{epoch}_{iteration}")]})





def train(opt):
    
    dataset = Flare7kpp_Pair_Loader(opt['dataset']['train'])

    logger.debug(f"Dataset length: {len(dataset)}")


    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=opt['num_workers'])    
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=opt['num_workers'])

    # make the generator
    generator = make_generator(opt['generator']['name'])
    generator_ema = make_generator(opt['generator']['name'])


    with torch.no_grad():
        for (gp, ep) in zip(generator.parameters(), generator_ema.parameters()):
            ep.data = gp.data.detach()


    discriminator = define_D(input_nc = 3 + 3, ndf = 64, n_layers_D = 3, num_D = 3, norm="instance", getIntermFeat=True)

    # optimizers
    G_optim = torch.optim.AdamW(generator.parameters(), lr=1e-4)
    D_optim = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)

    generator, discriminator = accelerator.prepare(generator, discriminator)
    generator_ema = accelerator.prepare(generator_ema)
    G_optim, D_optim = accelerator.prepare(G_optim, D_optim)
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('output/images', exist_ok=True)

    epochs = opt['epochs']
    loss = Losses()
    log = {}
    N = 0

    for epoch in range(epochs):
        logger.debug(f"Epoch: {epoch}")
        pbar = tqdm(train_loader)
        for data in pbar:

            # update the generator
            G_optim.zero_grad()
            generator.requires_grad_(True)
            discriminator.requires_grad_(False) 

            G_losses = loss.calc_G_losses(data, 
                                          generator, 
                                          discriminator)
            
            gen_wts = opt['loss_weights']['generator']

            G_loss = process_loss(log, G_losses, gen_wts)
            G_loss.backward()
            del G_losses
            G_optim.step()

            moving_average(generator, generator_ema,beta=0.999)

            D_optim.zero_grad()
            generator.requires_grad_(False)
            discriminator.requires_grad_(True)
            D_losses = loss.calc_D_losses(data, generator, 
                                          discriminator, 
                                          replay_pool)
            

            disc_wts = opt['loss_weights']['discriminator']

            D_loss = process_loss(log, D_losses, disc_wts)
            D_loss.backward()
            del D_losses
            D_optim.step()


            txt = ""
            N += 1
            if (N%100 == 0) or (N + 1 >= len(train_loader)):
                for i in range(3):
                    # test(epoch, N + i)
                    test(epoch, N+1, generator_ema, val_loader)
            for k in log:
                txt += f"{k}: {log[k]/N:.3e} "
            pbar.set_description(txt)
            
            if (N%1000 == 0) or (N + 1 >= len(train_loader)):
                import datetime
                out_file = f"epoch_{epoch}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.pt"
                out_file = os.path.join(checkpoint_dir, out_file)
                torch.save({"G": generator_ema.state_dict(), "D": discriminator.state_dict()}, out_file)
                print(f"Saved to {out_file}")



if __name__ == '__main__':
	
    opt = toml.load('opt.toml')

    wandb_opt = opt['wandb']    

    wandb_log = wandb.init(project='diffusion-flare', 
                           name = wandb_opt['name'], 
                           config=opt, 
                           notes = wandb_opt['notes'])
    
    train(opt)