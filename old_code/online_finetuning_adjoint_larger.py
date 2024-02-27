

"""

Only try to learn a simple neural network giving me the step length per time step: 

s_theta(x_t, t) - NN(t) A*(A x_t - y)


"""


import torch
import torchsde

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import yaml 

import torch 
import torch.nn as nn 

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

val_dataset = MNIST(root="./mnist_data",
                        train=False,
                        download=True,
                        transform = transforms.ToTensor()
                        )


x_gt = val_dataset[0][0].unsqueeze(0).to("cuda")

m = x_gt.numel()/2
A = torch.randn([int(m), int(x_gt.numel())])/torch.sqrt(torch.tensor(m))
A = A.to("cuda")

def Afwd(x, A=A):

    tmp = x.view(x.shape[0], -1)

    return torch.matmul(tmp, A.T)

def Abwd(y, A=A):

    tmp = torch.matmul(y, A)

    return tmp.view(tmp.shape[0], 1, 28, 28)

y = Afwd(x_gt)
y_noise = y + 0.01*torch.mean(y)*torch.rand_like(y)
ATy = Abwd(y_noise)

"""
fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(x_gt.cpu()[0,0,:,:])
ax1.set_title("GT")

ax2.imshow(ATy.cpu()[0,0,:,:])
ax2.set_title("A*(y)")

plt.show()
"""


class SDE(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'stratonovich'

    def __init__(self, model, sde, y_noise):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "stratonovich"

        self.model = model 
        self.sde = sde 
        self.cond_model = TinyUnet(
            marginal_prob_std=sde.marginal_prob_std, 
            time_embedding_dim=cfg_dict["model"]["time_embedding_dim"],
            max_period=cfg_dict["model"]["max_period"],
            in_channels=1,
            out_channels=cfg_dict["model"]["out_channels"],
            base_dim=16,
            dim_mults=[1,2])
        self.cond_model.to("cuda")
        self.cond_model.train() 

        self.y_noise = y_noise
    # Drift
    def f(self, t, y):
        y = y[:, :-1]

        #print(1.0 - t)
        ones_vec = torch.ones(y.shape[0], device=y.device)
        t = ones_vec * t
        
        s_pretrained = self.model(y.view(y.shape[0], 1, 28,28), 1.0 - t)
        
        cond = torch.repeat_interleave(self.y_noise,  dim=0, repeats=y.shape[0])
        log_grad = Abwd(Afwd(y.view(y.shape[0], 1, 28,28)) - cond)
        s_new = self.cond_model(log_grad, 1.0 - t)

        s = s_pretrained + s_new

        drift, diffusion = self.sde.sde(y.view(y.shape[0], 1, 28,28), 1.0 - t)

        mu = drift - diffusion[:, None, None, None].pow(2)*s

        beta_t = self.sde.beta_0 + (1.0 - t) * (self.sde.beta_1 - self.sde.beta_0)
        f_sq = beta_t.unsqueeze(1)*(s_new ** 2).sum(dim=(1,2,3)).unsqueeze(1)

        drift = -mu.view(y.shape[0], -1)
        
        return torch.cat([drift, f_sq], dim=1)

    # Diffusion
    def g(self, t, y):
        y = y[:, :-1]
        ones_vec = torch.ones(y.shape[0], device=y.device)
        t = ones_vec * t
        drift, diffusion = self.sde.sde(y.view(y.shape[0], 1, 28,28), 1.0 - t)

        diffusion_rep = diffusion[:,None].repeat(1, y.shape[-1])
        return torch.cat([diffusion_rep, torch.zeros((y.shape[0], 1), device=y.device)], dim=1)



sde_model = SDE(model=model, sde=sde, y_noise=y_noise)

print(sde_model.parameters())

print("PARAMETERS IN SDE MODEL: ", sum([p.numel() for p in sde_model.parameters()]))
print("PARAMETERS IN pretrained model: ", sum([p.numel() for p in sde_model.model.parameters()]))
print("PARAMETERS IN finetune: ", sum([p.numel() for p in sde_model.cond_model.parameters()]))


batch_size = 12

cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)

t_size = 200

optimizer = torch.optim.Adam(sde_model.cond_model.parameters(), lr=1e-3)

x_target = x_gt.repeat(batch_size, 1, 1, 1)

import time 
print(x_target.shape)
for i in range(1000):
    optimizer.zero_grad()

    y0 = torch.randn((batch_size, 784)).to("cuda")
    y0 = torch.cat([y0, torch.zeros((batch_size, 1), device=y0.device)], dim=1)
    bm = torchsde.BrownianInterval(t0=0.0, t1=1.0 - 1.e-3, size=(batch_size, 784 + 1), device='cuda')

    ts = torch.linspace(0, 1 - 1.e-3, t_size).to("cuda")

    t_start = time.time()
    #ys, logpq = torchsde.sdeint_adjoint(sde_model, y0, ts, method='euler', logqp=True)
    #ys = torchsde.sdeint_adjoint(sde_model, y0, ts, method='euler',adjoint_method="euler", dt=0.01)
    ys = torchsde.sdeint_adjoint(sde_model, y0, ts, 
                    method="heun",
                    #adjoint_method="adjoint_reversible_heun", 
                    dt=ts[1] - ts[0], bm=bm,
                    adjoint_params=sde_model.cond_model.parameters())

    #print(logpq.shape)
    ys_img = ys[-1, :, :-1]
    ys_img = ys_img.view(batch_size, 1, 28, 28)

    f_sq = ys[-1, :, -1]
    #loss = torch.sum(logpq**2) + 1/2*torch.sum((Afwd(ys) - y_noise)**2)
    loss_data = 1/2*torch.mean(torch.sum((Afwd(ys_img) - cond)**2, dim=1)) #1/2 * torch.mean(torch.sum((ys_img - x_target)**2, dim=(1,2,3)))
    print(loss_data, torch.mean(f_sq))
    loss = loss_data + torch.mean(f_sq) 
    print(loss.item())
    loss.backward()

    optimizer.step()
    t_end = time.time() 
    print("TIME FOR ONE GRADIENT STEP: ", t_end - t_start, "s")
    fig, (ax1, ax2) = plt.subplots(1,2)


    img_grid = make_grid(ys_img.cpu(), n_row=4)
    ax1.set_title(f"loss: {loss.item()}")
    ax1.imshow(x_target[0,0,:,:].cpu().numpy(), cmap="gray")
    ax2.imshow(img_grid[0,:,:].numpy(), cmap="gray")
    plt.savefig(f"results_adjoint/iter_{i}.png")

    plt.close()

    """
    if i % 5 == 0:# and i > 0:
        with torch.no_grad():
            y0 = torch.randn((batch_size, 784)).to("cuda")
            y0 = torch.cat([y0, torch.zeros((batch_size, 1), device=y0.device)], dim=1)

            ts = torch.linspace(0, 1 - 1.e-3, t_size).to("cuda")
            #ys, logpq = torchsde.sdeint_adjoint(sde_model, y0, ts, method='euler', logqp=True)
            #ys = torchsde.sdeint_adjoint(sde_model, y0, ts, method='euler',adjoint_method="euler", dt=0.01)
            ys = torchsde.sdeint(sde_model, y0, ts, method='euler',dt=ts[1] - ts[0], bm=bm)

            #print(logpq.shape)
            ys = ys[-1, :, :-1]
            ys = ys.view(batch_size, 1, 28, 28)

        img_grid = make_grid(ys.cpu(), n_row=4)
        #print(x_mean.shape, img_grid.shape)
        plt.figure()
        plt.imshow(img_grid[0,:,:].numpy(), cmap="gray")
        plt.show()

    """