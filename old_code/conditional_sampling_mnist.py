



"""
Simple conditional sampler 

s_theta(x_t, t) - lambda_t A*(A x_t - y)


"""

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

        self.cond = cond
    # Drift
    def f(self, t, y):

        #print(1.0 - t)
        ones_vec = torch.ones(y.shape[0], device=y.device)
        t = ones_vec * t
        
        drift, diffusion = self.sde.sde(y.view(y.shape[0], 1, 28,28), 1.0 - t)

        s_pretrained = self.model(y.view(y.shape[0], 1, 28,28), 1.0 - t)
    
        s_new = - 4*Abwd(Afwd(y.view(y.shape[0], 1, 28,28)) - self.cond)/torch.max(self.model.marginal_prob_std(1.0 - t)[:, None, None, None], torch.tensor(0.05,device=y.device))
        print(torch.linalg.norm(s_pretrained), torch.linalg.norm(s_new), self.model.marginal_prob_std(1.0 - t)[0])
        s = s_pretrained + s_new


        mu = drift - diffusion[:, None, None, None].pow(2)*s

        drift = -mu.view(y.shape[0], -1)

        return drift

    # Diffusion
    def g(self, t, y):
        ones_vec = torch.ones(y.shape[0], device=y.device)
        t = ones_vec * t
        drift, diffusion = self.sde.sde(y.view(y.shape[0], 1, 28,28), 1.0 - t)

        diffusion_rep = diffusion[:,None].repeat(1, y.shape[-1])
        return diffusion_rep

batch_size = 32

cond = torch.repeat_interleave(y_noise,  dim=0, repeats=batch_size)
sde_model = SDE(model=model, sde=sde, cond=cond)

y0 = torch.randn((batch_size, 784)).to("cuda")
t_size = 500
ts = torch.linspace(0, 1 - 1.e-3, t_size).to("cuda")
# Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
# ys will have shape (t_size, batch_size, state_size)

with torch.no_grad():
    ys = torchsde.sdeint(sde_model, y0, ts, method='euler', dt = ts[1] - ts[0])

print(ys.shape)

ys = ys[-1, :, :]
ys = ys.view(ys.shape[0], 1, 28, 28).cpu()


img_grid = make_grid(ys, n_row=4)
print(img_grid.shape)
#print(x_mean.shape, img_grid.shape)

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(x_gt[0,0,:,:].cpu().numpy(), cmap="gray")

ax2.imshow(img_grid[0,:,:].numpy(), cmap="gray")
plt.show()
