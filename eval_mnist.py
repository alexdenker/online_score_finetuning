

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


sampler = BaseSampler(
        score=model, 
        sde=sde,
        sampl_fn=Euler_Maruyama_sde_predictor,
        sample_kwargs={"batch_size": cfg_dict["sampling"]["batch_size"], 
                        "num_steps": 2000, #,cfg_dict["sampling"]["num_steps"],
                        "im_shape": [1,28,28],
                        "eps": cfg_dict["sampling"]["eps"] },
        device="cuda")

x_mean = sampler.sample().cpu()

x_mean = torch.clamp(x_mean.cpu(), 0, 1)

img_grid = make_grid(x_mean, n_row=4)
plt.figure()
plt.imshow(img_grid[0,:,:].numpy(), cmap="gray")
plt.show()