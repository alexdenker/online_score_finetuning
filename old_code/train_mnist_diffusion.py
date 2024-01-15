

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

from models.guided_diffusion.tiny_unet import TinyUnet
from models.diffusion import Diffusion
from htransform.sampler import DDIMSampler

cfg_dict = { 
    "model":
    {"time_embedding_dim": 256,
     "in_channels": 1,
     "out_channels": 1,
     "base_dim": 64,
     "dim_mults": [2,4]},
    "diffusion":
    {"beta_schedule": "linear",
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "num_diffusion_timesteps": 1000,
    },
    "training":
    {"num_epochs": 100,
     "batch_size": 128,
     "lr": 1e-4},
    "sampling": 
    {"num_steps": 250,
    "eta": 0.8,
    "batch_size": 16}
}


model = TinyUnet(timesteps=cfg_dict["diffusion"]["num_diffusion_timesteps"],
            time_embedding_dim=cfg_dict["model"]["time_embedding_dim"],
            in_channels=cfg_dict["model"]["in_channels"],
            out_channels=cfg_dict["model"]["out_channels"],
            base_dim=cfg_dict["model"]["base_dim"],
            dim_mults=cfg_dict["model"]["dim_mults"])
model.to("cuda")

train_dataset = MNIST(root="./mnist_data",
                        train=True,
                        download=True,
                        transform = transforms.ToTensor()
                        )


sde = Diffusion(beta_schedule=cfg_dict["diffusion"]["beta_schedule"],
            beta_start=cfg_dict["diffusion"]["beta_start"],
            beta_end=cfg_dict["diffusion"]["beta_end"],
            num_diffusion_timesteps=cfg_dict["diffusion"]["num_diffusion_timesteps"]
            )


train_dl = DataLoader(train_dataset, batch_size=cfg_dict["training"]["batch_size"])

optimizer = torch.optim.Adam(model.parameters(), lr=cfg_dict["training"]["lr"])

log_dir = "/localdata/AlexanderDenker/score_based_baseline/MNIST"

with open(os.path.join(log_dir, "report.yaml"), "w") as file:
    yaml.dump(cfg_dict, file)


for epoch in range(cfg_dict["training"]["num_epochs"]):
    print("Epoch: ", epoch+1)
    model.train()
    for i, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        optimizer.zero_grad() 
        x = batch[0].to("cuda")

        random_t = torch.randint(1, sde.num_diffusion_timesteps, (x.shape[0],), device=x.device)
        z = torch.randn_like(x)
        alpha_t = sde.alpha(random_t).view(-1, 1, 1, 1)

        perturbed_x = alpha_t.sqrt()*x + (1 - alpha_t).sqrt()*z

        zhat = model(perturbed_x, random_t)

        loss = torch.mean(torch.sum((z - zhat) ** 2, dim=(1, 2, 3)))
        print(loss.item())
        loss.backward()

        optimizer.step()

    model.eval() 

    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

    ddim = DDIMSampler(model=model,
                        diffusion=sde,
                        device=x.device, 
                        sample_kwargs = {"num_steps": cfg_dict["sampling"]["num_steps"],
                                                   "batch_size": cfg_dict["sampling"]["batch_size"],
                                                   "im_shape": x.shape[1:],
                                                   "eta": cfg_dict["sampling"]["eta"]}
                                )

    x_mean = ddim.sample().cpu()
    x_mean = torch.clamp(x_mean, 0, 1)

    save_image(x_mean, os.path.join(log_dir, f"sample_at_{epoch}.png"),nrow=4)
    #plt.figure()
    #plt.imshow(x_mean[0,0,:,:].cpu().numpy(), cmap="gray")
    #plt.show() 

    #img_grid = make_grid(x_mean, n_row=4)
    #print(x_mean.shape, img_grid.shape)
    #plt.figure()
    #plt.imshow(img_grid[0,:,:].numpy(), cmap="gray")
    #plt.show()