

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
    sde_type = 'ito'

    def __init__(self, model, sde, cond):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        self.model = model 
        self.sde = sde 
        self.time_model = nn.Sequential(nn.Linear(1, 32),
                                    nn.ReLU(), 
                                    nn.Linear(32,1))
        for p in self.time_model[-1].parameters():
            p.detach().zero_()

        self.time_model.to("cuda")
        self.time_model.train() 

        self.cond = cond
    # Drift
    def f(self, t, y):
        y = y[:, :-1]

        #print(1.0 - t)
        ones_vec = torch.ones(y.shape[0], device=y.device)
        t = ones_vec * t
        
        s_pretrained = self.model(y.view(y.shape[0], 1, 28,28), 1.0 - t)
        
        time_cond = self.time_model(1.0 - t.unsqueeze(1))
        s_new = - time_cond[:,None,None]*Abwd(Afwd(y.view(y.shape[0], 1, 28,28)) - self.cond)/torch.max(self.model.marginal_prob_std(1.0 - t)[:, None, None, None], torch.tensor(0.05,device=y.device))

        s = s_pretrained + s_new

        drift, diffusion = self.sde.sde(y.view(y.shape[0], 1, 28,28), 1.0 - t)

        mu = drift - diffusion[:, None, None, None].pow(2)*s

        f_sq = (time_cond ** 2).sum(dim=1, keepdim=True)
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

batch_size = 128

cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)

sde_model = SDE(model=model, sde=sde, cond=cond)
t_size = 200

optimizer = torch.optim.Adam(sde_model.time_model.parameters(), lr=1e-3)

x_target = x_gt.repeat(batch_size, 1, 1, 1)


print(x_target.shape)
for i in range(1000):
    optimizer.zero_grad()

    y0 = torch.randn((batch_size, 784)).to("cuda")
    y0 = torch.cat([y0, torch.zeros((batch_size, 1), device=y0.device)], dim=1)
    bm = torchsde.BrownianInterval(t0=0.0, t1=1.0 - 1.e-3, size=(batch_size, 784 + 1), device='cuda')

    ts = torch.linspace(0, 1 - 1.e-3, t_size).to("cuda")

    #ys, logpq = torchsde.sdeint_adjoint(sde_model, y0, ts, method='euler', logqp=True)
    #ys = torchsde.sdeint_adjoint(sde_model, y0, ts, method='euler',adjoint_method="euler", dt=0.01)
    ys = torchsde.sdeint_adjoint(sde_model, y0, ts, method='euler',adjoint_method="euler", dt=ts[1] - ts[0], bm=bm)

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
    
    fig, (ax1, ax2) = plt.subplots(1,2)


    img_grid = make_grid(ys_img.cpu(), n_row=4)
    ax1.set_title(f"loss: {loss.item()}")
    ax1.imshow(x_target[0,0,:,:].cpu().numpy(), cmap="gray")
    ax2.imshow(img_grid[0,:,:].numpy(), cmap="gray")
    plt.savefig(f"adjoint_test_imgs_fwd/iter_{i}.png")

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