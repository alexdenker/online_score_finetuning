"""
File from Francisco Vargas.

"""

import yaml 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import torch 
from torchvision.datasets import MNIST
from torchvision import transforms 
from tqdm import tqdm 


from models.guided_diffusion.tiny_unet import TinyUnet
from optimal_control.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule

def _schedule_jump(num_steps, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, num_steps - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = num_steps
    time_steps = []
    while t >= 1:
        t = t - 1
        time_steps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                time_steps.append(t)
    time_steps.append(-1)
    _check_times(time_steps, -1, num_steps)

    return time_steps

def _check_times(times, t_0, num_steps):
    assert times[0] > times[1], (times[0], times[1])

    assert times[-1] == -1, times[-1]

    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= num_steps, (t, num_steps)


def get_loss_ddim(condition: torch.Tensor, pretrained_model, cond_model, sde, A, AT, normed=False):
    
    condition = condition

    def loss(condition=condition):

        xi = torch.randn((1,1,28,28)).to("cuda")
        device = "cuda"

        # network_ref = network_cond.unconditional
        tdist = torch.distributions        

        ATy = AT(condition).to("cuda")

        num_steps = 100
        skip = sde.num_timesteps // num_steps

        time_steps = _schedule_jump(num_steps, 1,1)
        time_pairs = list(
                (i * skip, j * skip if j > 0 else -1)
                for i, j in zip(time_steps[:-1], time_steps[1:])
            )

        loss = 0.0
        for i, j in tqdm(time_pairs):
            ones_vec = torch.ones(len(xi), device=device)

            if j<0: j=0

            time_step = (ones_vec * i, ones_vec * j)  # (t, tminus1)

            t = time_step[0]
            tminus1 = time_step[1]

            def _cond_model(x, t, **kwargs):
                model_inp = torch.cat([x, ATy], dim=1)

                return model(x, t) + cond_model(model_inp, t)
            cond_dict = sde.ddim_sample(_cond_model, xi, t.long(), tminus1.long(), eta=0.8)

            xi = cond_dict["sample"]

            #loss += torch.sum(cond_dict["eps"]**2)
        
            if j == 0:
                break

        condition = condition.to("cuda")

        loss_0 = 1/2*torch.sum((Afwd(xi) - condition)**2)    #likelihood.loss(xi, condition)
        print("loss0: ", loss_0)
        loss += loss_0 
        print("loss: ", loss)

        return loss

    return loss

base_path = "/localdata/AlexanderDenker/score_based_baseline/MNIST"

with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)

model = TinyUnet(timesteps=cfg_dict["diffusion"]["num_diffusion_timesteps"],
            time_embedding_dim=cfg_dict["model"]["time_embedding_dim"],
            in_channels=cfg_dict["model"]["in_channels"],
            out_channels=cfg_dict["model"]["out_channels"],
            base_dim=cfg_dict["model"]["base_dim"],
            dim_mults=cfg_dict["model"]["dim_mults"])
model.to("cuda")
model.load_state_dict(torch.load(os.path.join(base_path, "model.pt")))
model.eval() 

val_dataset = MNIST(root="./mnist_data",
                        train=False,
                        download=True,
                        transform = transforms.ToTensor()
                        )

betas = get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=cfg_dict["diffusion"]["num_diffusion_timesteps"])
sde = GaussianDiffusion(betas)

x = val_dataset[0][0].unsqueeze(0)

m = x.numel()/2

A = torch.randn([int(m), int(x.numel())])/torch.sqrt(torch.tensor(m))

def Afwd(x, A=A):

    tmp = x.view(x.shape[0], -1)

    return torch.matmul(tmp, A.T.to(x.device))

def Abwd(y, A=A):

    tmp = torch.matmul(y, A.to(y.device))

    return tmp.view(tmp.shape[0], 1, 28, 28)





y = Afwd(x)
y_noise = y + 0.01*torch.mean(y)*torch.rand_like(y)
ATy = Abwd(y_noise).to("cuda")

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(x.cpu()[0,0,:,:])
ax1.set_title("GT")

ax2.imshow(ATy.cpu()[0,0,:,:])
ax2.set_title("A*(y)")

plt.show()


cond_model = TinyUnet(timesteps=cfg_dict["diffusion"]["num_diffusion_timesteps"],
            time_embedding_dim=cfg_dict["model"]["time_embedding_dim"],
            in_channels=2,
            out_channels=cfg_dict["model"]["out_channels"],
            base_dim=32,
            dim_mults=[1,2])
cond_model.train()
cond_model.to("cuda")

losses = []
optimizer = torch.optim.Adam(cond_model.parameters(), lr=1e-3)
loss_fn = get_loss_ddim(y_noise, normed=True, pretrained_model=model, cond_model=cond_model, sde=sde, A=Afwd, AT=Abwd)

for train_step_idx in range(1000):
    optimizer.zero_grad()
    loss_value = loss_fn()
    loss_value.backward()
    optimizer.step()
    print(loss_value)
    losses.append(loss_value)

    if train_step_idx % 20 == 0 and train_step_idx > 0:
        # unconditional sampling 
        with torch.no_grad():
            xi = torch.randn((1,1,28,28)).to("cuda")
            device = "cuda"
            
            num_steps = 100
            skip = sde.num_timesteps // num_steps

            time_steps = _schedule_jump(num_steps, 1,1)
            time_pairs = list(
                    (i * skip, j * skip if j > 0 else -1)
                    for i, j in zip(time_steps[:-1], time_steps[1:])
                )
            for i, j in tqdm(time_pairs):
                ones_vec = torch.ones(len(xi), device=device)
                if j<0: j=0

                time_step = (ones_vec * i, ones_vec * j)  # (t, tminus1)

                t = time_step[0]
                tminus1 = time_step[1]

                def _cond_model(x, t, **kwargs):
                    model_inp = torch.cat([x, ATy], dim=1)

                    return model(x, t) + cond_model(model_inp, t)
                ref_dict = sde.ddim_sample(_cond_model, xi, t.long(), tminus1.long(), eta=0.8)

                xi = ref_dict["sample"]
                
                if j==0:
                    break

        plt.figure()
        plt.imshow(xi.detach().cpu().numpy()[0,0,:,:])
        plt.title("new sample")
        plt.show()