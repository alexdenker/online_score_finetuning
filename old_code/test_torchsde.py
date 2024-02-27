


import torch
import torchsde

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 

import torch 

from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm 

from src import TinyUnet, VPSDE, BaseSampler, Euler_Maruyama_sde_predictor


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


print("num params of model: ", sum([p.numel() for p in model.parameters()]))


class SDE(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito' #'ito' #'stratonovich' # stratonovich

    def __init__(self, model, sde):
        super().__init__()
        
        self.model = model 
        self.sde = sde 
        
        self.noise_type = "diagonal"
        self.sde_type = 'ito' #'ito' #"stratonovich" # stratonovich
    # Drift
    def f(self, t, y):
        print(t)
        ones_vec = torch.ones(y.shape[0], device=y.device)
        t = ones_vec * t

        s = self.model(y.view(y.shape[0], 1, 28,28), 1. - t) 

        drift, diffusion = self.sde.sde(y.view(y.shape[0], 1, 28,28), 1. - t)

        mu = drift - diffusion[:, None, None, None].pow(2)*s

        return -mu.view(y.shape[0], -1)

    # Diffusion
    def g(self, t, y):
        ones_vec = torch.ones(y.shape[0], device=y.device)
        t = ones_vec * t
        drift, diffusion = self.sde.sde(y.view(y.shape[0], 1, 28,28), 1. - t)

        diffusion_rep = diffusion[:,None].repeat(1, y.shape[-1])
        return diffusion_rep


batch_size = 32

sde_model = SDE(model=model, sde=sde)
y0 = torch.randn((batch_size, 784)).to("cuda")
t_size = 2000
ts = torch.linspace(0, 1 - 1.e-2, t_size).to("cuda")
# Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
# ys will have shape (t_size, batch_size, state_size)

with torch.no_grad():
    #ys = torchsde.sdeint(sde_model, y0, ts, method='reversible_heun', dt = ts[1] - ts[0])
    ys = torchsde.sdeint(sde_model, y0, ts, method='euler', dt = ts[1] - ts[0])

print(ys.shape)

ys = ys[-1, :, :]
ys = ys.view(ys.shape[0], 1, 28, 28).cpu()


img_grid = make_grid(ys, n_row=4)
print(img_grid.shape)
#print(x_mean.shape, img_grid.shape)
plt.figure()
plt.imshow(img_grid[0,:,:].numpy(), cmap="gray")
plt.show()