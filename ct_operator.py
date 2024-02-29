

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

from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm 

from src import Tomography

cfg = {
    'batch_size': 64,
    'time_steps': 75,
    'lr': 5e-4,
    'num_steps':  1000,
    'log_img_freq': 5,
    'angles': 3,
    'rel_noise': 0.05,
    'use_tweedie': True,
    'clip_gradient': False,
    'clip_value': 1.0, 
}


base_path = "model_weights"

with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)


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


print("SHAPE OF MEASUREMENTS: ", y_noise.shape)
print("min/max of y: ", y_noise.min(), y_noise.max() )

print("|| A x - y ||^2 / ||y||^2 = ", 1800./(y_noise**2).sum())
print("|| y - y_noise ||^2 / ||y_noise||^2 = ", ((y - y_noise)**2).sum()/(y_noise**2).sum())


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

plt.show()
