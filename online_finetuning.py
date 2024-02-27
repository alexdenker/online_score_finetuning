

"""

Full backpropagation through the solver
"""

import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 
import time 

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
    'batch_size': 16,
    'time_steps': 40,
    'lr': 1e-4,
    'num_steps':  1000,
    'log_img_freq': 5,
    'angles': 3,
    'rel_noise': 0.05,
    'use_tweedie': True
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
    
    def forward(self, ts, xT):
        """
        Implement EM solver

        """
        x_t = [xT] 
        kldiv = torch.zeros(1).to(xT.device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0 
            dW = torch.randn_like(xT) * torch.sqrt(dt.abs())
            ones_vec = torch.ones(xT.shape[0], device=y.device)
            t = ones_vec * t0
        
            s_pretrained = self.model(x_t[-1], t)

            cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=y.shape[0])

            with torch.no_grad():
                if cfg["use_tweedie"]:
                    marginal_prob_mean = self.sde.marginal_prob_mean_scale(t)
                    marginal_prob_std = self.sde.marginal_prob_std(t)

                    x0hat = (x_t[-1] + marginal_prob_std[:,None,None,None]**2*s_pretrained)/marginal_prob_mean[:,None,None,None]
                    log_grad = forward_op.A_adjoint(forward_op.A(x0hat) - cond)

                else:
                    log_grad = forward_op.A_adjoint(forward_op.A(x_t[-1]) - cond)
                
            log_grad_scaling = self.time_model(t.unsqueeze(-1)).view(t.shape[0], 1, 28, 28)
            s_new = self.cond_model(torch.cat([log_grad, x_t[-1]], dim=1), t) 
            s_new = s_new - log_grad_scaling*(1. - t)[:,None,None,None]*log_grad # additional (1 - t) scaling (maybe not neccessary)

            s = s_pretrained + s_new

            drift, diffusion = self.sde.sde(x_t[-1], t)

            f_t = drift - diffusion[:, None, None, None].pow(2)*s

            f_sq = (s_new ** 2).sum(dim=(1,2,3)).unsqueeze(1)
            kldiv = kldiv + dt.abs() * f_sq * diffusion.pow(2)

            g_t = diffusion[:, None, None, None]
            x_t.append(x_t[-1] + f_t * dt + g_t * dW)

        return x_t, kldiv



wandb_kwargs = {
        "project": "online_finetuning",
        "entity": "alexanderdenker",
        "config": cfg,
        "name": None,
        "mode": "online" ,
        "settings": wandb.Settings(code_dir="/localdata/AlexanderDenker/online_finetuning"),
        "dir": "/localdata/AlexanderDenker/online_finetuning",
    }
with wandb.init(**wandb_kwargs) as run:

    batch_size = cfg["batch_size"]
    sde_model = CondSDE(model=model, sde=sde, y_noise=y_noise)
    t_size = cfg["time_steps"]
    optimizer = torch.optim.Adam(list(sde_model.cond_model.parameters()) + list(sde_model.time_model.parameters()), lr=cfg["lr"])

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

        xT = torch.randn((batch_size, 1, 28, 28)).to("cuda")

        ts = np.linspace(1e-3, 1., t_size)[::-1].copy()
        ts = torch.from_numpy(ts).to("cuda")**2

        time_start = time.time()
        
        xt, kldiv = sde_model.forward(ts, xT)

        #print("Solve forward SDE: ", time.time() - time_start, "s")
        #print(logpq.shape)

        cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)
        loss_data = 1/2*1/noise_level**2*torch.mean(torch.sum((forward_op.A(xt[-1]) - cond)**2, dim=(1,2,3)))  
        loss_kldiv = 0.5*kldiv.mean()
        loss = loss_data + loss_kldiv

        wandb.log(
                    {"train/loss": loss.item(), "step": i}
                ) 
        wandb.log(
                    {"train/loss_data_consistency": loss_data.item(), "step": i}
                ) 
        wandb.log(
                    {"train/loss_kldiv": loss_kldiv.item(), "step": i}
                ) 

        mse_target = torch.mean((xt[-1] - x_target)**2)
        wandb.log(
                    {"train/mse_to_target": mse_target.item(), "step": i}
                ) 
        time_start = time.time() 
        loss.backward()
        #print("Calculate Gradient, adjoint SDE: ", time.time() - time_start, "s")

        optimizer.step()

        if i % cfg["log_img_freq"] == 0:
            
            img_grid = make_grid(xt[-1].cpu(), nrow=4)

            plt.figure()
            plt.imshow(img_grid[0,:,:].numpy(), cmap="gray")
            wandb.log({f"samples": wandb.Image(plt)})

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

        