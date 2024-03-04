

"""
Implementation of the VarGrad loss. Change the direction of the SDE: going from 0 -> T
with x0 ~ N(0, I) and xT ~ data distribution
 
"""

import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 
import time 
import math 

import torch 
import torch.nn as nn 

import wandb

from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm 

from src import TinyUnet, VPSDE, Tomography

cfg = {
    'batch_size': 32,
    'time_steps': 70,
    'lr': 1e-3,
    'num_steps':  1000,
    'log_img_freq': 5,
    'angles': 3,
    'rel_noise': 0.05,
    'use_tweedie': True,
    'clip_gradient': True,
    'clip_value': 1.0, 
}


base_path = "model_weights"

with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)

sde = VPSDE(beta_min=cfg_dict["diffusion"]["beta_min"], 
            beta_max=cfg_dict["diffusion"]["beta_max"]
            )

model = TinyUnet(
            marginal_prob_std=sde.marginal_prob_std, 
            time_embedding_dim=cfg_dict["model"]["time_embedding_dim"],
            max_period=cfg_dict["model"]["max_period"],
            in_channels=cfg_dict["model"]["in_channels"],
            out_channels=cfg_dict["model"]["out_channels"],
            base_dim=cfg_dict["model"]["base_dim"],
            dim_mults=cfg_dict["model"]["dim_mults"])
model.load_state_dict(torch.load("model_weights/model.pt"))
model.to("cuda")
model.eval() 

val_dataset = MNIST(root="./mnist_data",
                        train=False,
                        download=True,
                        transform = transforms.ToTensor()
                        )


x_gt = val_dataset[9][0].unsqueeze(0).to("cuda")

forward_op = Tomography(img_width=28, angles=cfg["angles"], device="cuda")

with torch.no_grad():

    y = forward_op.A(x_gt)
    print(torch.mean(y.abs()))
    noise_level =  cfg["rel_noise"]*torch.mean(y.abs())
    print(noise_level, 1/noise_level**2)
    y_noise = y + noise_level*torch.rand_like(y)
    ATy = forward_op.A_adjoint(y_noise)

    x_fbp = forward_op.A_dagger(y_noise)



class CondSDE(torch.nn.Module):
    def __init__(self, model, sde, y_noise):
        super().__init__()

        self.model = model 
        self.sde = sde 
        self.cond_model = TinyUnet(
            marginal_prob_std=sde.marginal_prob_std, 
            time_embedding_dim=cfg_dict["model"]["time_embedding_dim"],
            max_period=cfg_dict["model"]["max_period"],
            in_channels=2,
            out_channels=cfg_dict["model"]["out_channels"],
            base_dim=16,
            dim_mults=[1,2])
        self.cond_model.to("cuda")
        self.cond_model.train() 

        self.time_model = nn.Linear(1, 784)
        self.time_model.to("cuda")
        self.time_model.weight.data.fill_(0.0)
        self.time_model.bias.data.fill_(0.2)

        self.y_noise = y_noise
    
    def forward(self, ts, x0):
        """
        Implement EM solver

        """
        x_t = [x0] 
        #print("time steps: ", ts)
        kldiv_term1 = torch.zeros(1).to(x0.device)
        kldiv_term2 = torch.zeros(1).to(x0.device)
        kldiv_term3 = torch.zeros(1).to(x0.device)

        for t0, t1 in zip(ts[:-1], ts[1:]):
            #print(t0, t1)
            dt = t1 - t0 
            #print(dt)
            dW = torch.randn_like(x0) * torch.sqrt(dt)
            ones_vec = torch.ones(x0.shape[0], device=y.device)
            t = ones_vec * t0
            #print("Time step: ", t0, " to model: ", 1-t0)
            with torch.no_grad():
                s_pretrained = self.model(x_t[-1], 1 - t)

            cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=y.shape[0])

            with torch.no_grad():
                if cfg["use_tweedie"]:
                    marginal_prob_mean = self.sde.marginal_prob_mean_scale(1-t)
                    marginal_prob_std = self.sde.marginal_prob_std(1-t)

                    x0hat = (x_t[-1] + marginal_prob_std[:,None,None,None]**2*s_pretrained)/marginal_prob_mean[:,None,None,None]
                    log_grad = forward_op.A_adjoint(forward_op.A(x0hat) - cond)

                else:
                    log_grad = forward_op.A_adjoint(forward_op.A(x_t[-1]) - cond)
                
            log_grad_scaling = self.time_model(1 - t.unsqueeze(-1)).view(t.shape[0], 1, 28, 28)
            s_new = self.cond_model(torch.cat([log_grad, x_t[-1]], dim=1), 1 - t) 
            s_new = s_new - log_grad_scaling*t[:,None,None,None]*log_grad # additional t scaling (maybe not neccessary)

            s = s_pretrained + s_new

            drift, diffusion = self.sde.sde(x_t[-1], 1 - t) # diffusion = sqrt(beta)
            # drift = - 0.5 beta x
            f_t =  - drift + diffusion[:, None, None, None].pow(2)*s

            #print("s_new shape: ", s_new.shape)


            f_sq = (s_new ** 2).sum(dim=(1,2,3))
            g_f = (s_new * s_new.detach()).sum(dim=(1,2,3))
            f_w = (s_new * dW).sum(dim=(1,2,3))
            #print("f_sq.shape: ", f_sq.shape, " g_f.shape: ", g_f.shape, " f_w.shape: ", f_w.shape)
            #print(dt.shape, diffusion.shape, diffusion.shape)
            kldiv_term1 = kldiv_term1 - 0.5 * f_sq * diffusion.pow(2) * dt
            kldiv_term2 = kldiv_term2 + diffusion.pow(2) * dt * g_f 
            kldiv_term3 = kldiv_term3 + diffusion * f_w
            #print(kldiv_term1.shape)
            #print(kldiv_term1.mean(), kldiv_term2.mean(), kldiv_term3.mean())
            g_t = diffusion[:, None, None, None]
            x_new = x_t[-1] + f_t * dt + g_t * dW
            x_t.append(x_new.detach())
            
        return x_t, kldiv_term1, kldiv_term2, kldiv_term3



wandb_kwargs = {
        "project": "online_finetuning",
        "entity": "alexanderdenker",
        "config": cfg,
        "name": None,
        "mode": "online", #"disabled", #"online" ,
        "settings": wandb.Settings(code_dir="/localdata/AlexanderDenker/online_finetuning"),
        "dir": "/localdata/AlexanderDenker/online_finetuning",
    }
with wandb.init(**wandb_kwargs) as run:

    batch_size = cfg["batch_size"]
    sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)
    t_size = cfg["time_steps"]

    k = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    optimizer = torch.optim.Adam(list(sde_model.cond_model.parameters()) + list(sde_model.time_model.parameters()), lr=cfg["lr"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    x_target = x_gt.repeat(batch_size, 1, 1, 1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14,6))

    ax1.set_title("GT")
    ax1.imshow(x_gt[0,0,:,:].cpu(), cmap="gray")
    ax1.axis("off")

    ax2.set_title("y")
    ax2.imshow(y_noise[0,0,:,:].cpu(), cmap="gray")
    ax2.axis("off")

    ax3.set_title("A^T(x)")
    ax3.imshow(ATy[0,0,:,:].cpu(), cmap="gray")
    ax3.axis("off")

    ax4.set_title("FBP")
    ax4.imshow(x_fbp[0,0,:,:].cpu(), cmap="gray")
    ax4.axis("off")
    wandb.log({f"data": wandb.Image(plt)})

    plt.close()



    for i in tqdm(range(cfg["num_steps"])):
        optimizer.zero_grad()

        x0 = torch.randn((batch_size, 1, 28, 28)).to("cuda")

        ts = np.linspace(0, 1. - 1e-3, t_size).copy()
        ts = torch.from_numpy(ts).to("cuda").sqrt()
        time_start = time.time()
        
        xt, kldiv_term1, kldiv_term2, kldiv_term3 = sde_model.forward(ts, x0)

        #print("Solve forward SDE: ", time.time() - time_start, "s")
        #print(logpq.shape)

        """
        print(xt[-1].shape)
        y_sim = forward_op.A(xt[-1])
        print("y_sim: ", y_sim.shape)

        fig, (ax1, ax2, ax3) = plt.subplots(1,3)

        im = ax1.imshow(xt[-1].cpu()[0,0,:,:], cmap="gray")
        fig.colorbar(im, ax=ax1)


        im = ax2.imshow(y_sim.cpu()[0,0,:,:])
        fig.colorbar(im, ax=ax2)

        
        im = ax3.imshow(y_noise.cpu()[0,0,:,:])
        fig.colorbar(im, ax=ax3)

        plt.show()
        """

        cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)
        data_fit = torch.sum((forward_op.A(xt[-1]) - cond)**2, dim=(1,2,3))
        loss_data = 1/2*1/noise_level**2*data_fit
        #print(loss_data.shape, loss_kldiv.shape)
        loss_kl = kldiv_term1 + kldiv_term2 + kldiv_term3
        loss = loss_data + kldiv_term1 + kldiv_term2 + kldiv_term3

        # original finetuning loss 
        original_loss = (loss_data - kldiv_term1).mean()

        #print(loss)
        loss = torch.mean((loss - k)**2)
        print("var loss: ", loss)

        wandb.log(
                    {"train/loss": loss.item(), "step": i}
                ) 
        wandb.log(
                    {"train/original_finetuning_loss": original_loss.item(), "step": i}
                )
        wandb.log(
                    {"train/loss_data_consistency": loss_data.mean().item(), "step": i}
                ) 
        wandb.log(
                    {"train/data_fit(without 1/sigma^2)": data_fit.mean().item(), "step": i}
                ) 
        wandb.log(
                    {"train/loss_kldiv_term1": kldiv_term1.mean().item(), "step": i}
                ) 

        wandb.log(
                    {"train/loss_kldiv_term1": kldiv_term1.mean().item(), "step": i}
                ) 

        wandb.log(
                    {"train/loss_kldiv_term2": kldiv_term2.mean().item(), "step": i}
                ) 

        wandb.log(
                    {"train/loss_kldiv_term3": kldiv_term3.mean().item(), "step": i}
                ) 

        wandb.log(
                    {"train/loss_kldiv": loss_kl.mean().item(), "step": i}
                ) 
        wandb.log(
                    {"train/k": k.item(), "step": i}
                ) 
        mse_target = torch.mean((xt[-1] - x_target)**2)
        wandb.log(
                    {"train/mse_to_target": mse_target.item(), "step": i}
                ) 
        time_start = time.time() 
        loss.backward()
        #print("Calculate Gradient, adjoint SDE: ", time.time() - time_start, "s")
        if cfg['clip_gradient']:
            torch.nn.utils.clip_grad_norm_(sde_model.cond_model.parameters(), cfg['clip_value'])
            torch.nn.utils.clip_grad_norm_(sde_model.time_model.parameters(), cfg['clip_value'])
        optimizer.step()
        with torch.no_grad():
            k -= 0.005 * k.grad
        print(k)
        scheduler.step()
        wandb.log(
                    {"train/learning_rate": float(scheduler.get_last_lr()[0]), "step": i}
                )

        if i % cfg["log_img_freq"] == 0:
            
            img_grid = make_grid(xt[-1].cpu(), nrow=4)

            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,6))
            ax1.imshow(img_grid[0,:,:].numpy(), cmap="gray")
            ax1.set_title("samples")
            ax1.axis("off")

            mean_sample = xt[-1].cpu().mean(dim=0)
            #print(mean_sample.shape)
            ax2.imshow(mean_sample[0,:,:], cmap="gray")
            ax2.set_title("mean sample")
            ax2.axis("off")

            diff_to_mean = torch.mean((xt[-1].cpu() - mean_sample.unsqueeze(0))**2, dim=(1,2,3))
            diff_to_gt = torch.mean((x_target.cpu() - mean_sample)**2, dim=(1,2,3))

            #print(diff_to_mean)

            ax3.hist(diff_to_mean.ravel().numpy(), bins="auto", alpha=0.75)
            ax3.set_title("| x - x_mean|")
            ax3.vlines(diff_to_gt.numpy(), 0, 30, label="| x_mean - x_gt|", colors='r')
            ax3.legend()
            wandb.log({f"samples": wandb.Image(plt)})

            x_std = torch.std(xt[-1].cpu(), dim=0)
            #print("x_std: ", x_std.shape)
            #print("x_target: ", x_target.shape)
            #print("mean_sample: ", mean_sample.shape)
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(16,6))

            im = ax1.imshow(x_target.cpu()[0,0,:,:], cmap="gray")
            ax1.axis("off")
            ax1.set_title("Ground truth")
            fig.colorbar(im, ax=ax1)

            im = ax2.imshow(mean_sample[0,:,:], cmap="gray")
            ax2.axis("off")
            ax2.set_title("Mean Sample")
            fig.colorbar(im, ax=ax2)

            diff = torch.abs(x_target.cpu()[0,0,:,:] - mean_sample[0,:,:])
            im = ax3.imshow(diff, cmap="gray")
            ax3.axis("off")
            ax3.set_title("|x_gt - x_mean|")
            fig.colorbar(im, ax=ax3)

            im = ax4.imshow(x_std.numpy()[0,:,:], cmap="gray")
            ax4.axis("off")
            ax4.set_title("x_std")
            fig.colorbar(im, ax=ax4)

            ax5.hist(diff.numpy().ravel(), label="|x_gt - x_mean|", alpha=0.4, bins="auto")
            ax5.hist(x_std.numpy().ravel(), label="x_std", alpha=0.4, bins="auto")
            ax5.legend()
            wandb.log({f"validation": wandb.Image(plt)})

            """
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
            ax1.imshow(x_target[0,0,:,:].cpu().numpy(), cmap="gray")
            ax1.axis("off")
            ax2.imshow(img_grid[0,:,:].numpy(), cmap="gray")
            ax2.axis("off")
            #plt.savefig(f"results/val_iter_{i}.png")
            wandb.log({f"samples": wandb.Image(plt)})
            """
            plt.close()

        