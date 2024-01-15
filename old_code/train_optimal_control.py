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

def get_loss_is(condition: torch.Tensor, pretrained_model, cond_model, sde, A, AT, normed=False):
    
    condition = condition

    def loss(condition=condition):

        xi = torch.randn((1,1,28,28)).to("cuda")
        device = "cuda"
        loss = 0.0
        ln_is_w = 0.0

        # network_ref = network_cond.unconditional
        tdist = torch.distributions        

        ATy = AT(condition).to("cuda")

        for time in reversed(range(sde.num_timesteps)[::10]):
            print(time)
            batched_t = torch.ones((len(condition))).fill_(time).long().to(device)

            #eps = pretrained_model(xi, batched_t)

            #model_inp = torch.cat([xi, ATy], dim=1)
            #eps_cond = cond_model(model_inp, batched_t)

            with torch.no_grad():
                ref_dict = sde.p_sample(model, xi, batched_t)
            cond_dict = sde.p_sample(lambda x, t: cond_model(torch.cat([x, ATy], dim=1), t), xi, batched_t)

            mu_ref = ref_dict["mean"]
            log_var_ref = ref_dict["log_variance"]

            mu_cond = cond_dict["mean"]
            log_var_cond = cond_dict["log_variance"]
            #x0 = (xi - (1 - alpha_t).sqrt()*eps) / alpha_t.sqrt() 
            #x0_cond = (xi - (1 - alpha_t).sqrt()*eps_cond) / alpha_t.sqrt() 

            #mu_ref, var_ref, log_var_ref = p_mean_variance(x_start=x0, x=x_t, t=batched_i, sde=sde)
            scale_ref = (0.5 * log_var_ref).exp()

            #mu_cond, _, log_var_cond = p_mean_variance(x_start=x0_cond, x=x_t, t=batched_i, sde=sde)
            scale_cond = (0.5 * log_var_cond).exp()

            p_cond = tdist.Normal(mu_cond, scale_cond)
            p_ref = tdist.Normal(mu_ref, scale_ref)

            xi = ref_dict["sample"]

            loss_i = p_cond.log_prob(xi) - p_ref.log_prob(xi)
            loss += torch.sum(loss_i, dim=(1, 2, 3))
        
            # is weight could be different for a diffrent proposal
            lwi = loss_i
            ln_is_w += lwi
        
        condition = condition.to("cuda")
        print(xi.shape)
        plt.figure()
        plt.imshow(xi.detach().cpu().numpy()[0,0,:,:])
        plt.show()

        loss_0 = 1/2*torch.sum((Afwd(xi) - condition)**2)    #likelihood.loss(xi, condition)
        print("loss0: ", loss_0)
        loss += loss_0 
        print("loss: ", loss)
        # we cannot use logsumexp
        # so hoping this deals with numerical issues
        # However this is biased?
        weights = torch.exp(loss)

        if normed:  # this is biased 
            weights = torch.exp(lwi - lwi.max())
            weights = weights / weights.sum()
        
        loss_final = (weights * loss).mean()
        print("loss_final (weighted): ", loss_final)
        return loss_final

    return loss



def get_loss_ddim_v2(condition: torch.Tensor, pretrained_model, cond_model, sde, A, AT, normed=False):
    
    condition = torch.repeat_interleave(condition,  dim=0, repeats=10)
    print(condition.shape)
    def loss(condition=condition):

        xi = torch.randn((10,1,28,28)).to("cuda")
        device = "cuda"
        loss = 0.0
        ln_is_w = 0.0

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
        for i, j in tqdm(time_pairs):
            ones_vec = torch.ones(len(xi), device=device)

            if j<0: j=0

            time_step = (ones_vec * i, ones_vec * j)  # (t, tminus1)

            t = time_step[0]
            tminus1 = time_step[1]

            with torch.no_grad():
                ref_dict = sde.ddim_sample(model, xi, t.long(), tminus1.long(), eta=0.8)

            def _cond_model(x, t, **kwargs):
                model_inp = torch.cat([x, ATy], dim=1)

                with torch.no_grad():
                    m1 = model(x,t)

                return m1 + cond_model(model_inp, t)

            cond_dict = sde.ddim_sample(_cond_model, xi, t.long(), tminus1.long(), eta=0.8)

            mu_ref = ref_dict["mean"]
            log_var_ref = ref_dict["log_variance"]

            mu_cond = cond_dict["mean"]
            log_var_cond = cond_dict["log_variance"]
            
            scale_ref = (0.5 * log_var_ref).exp()
            scale_cond = (0.5 * log_var_cond).exp()

            p_cond = tdist.Normal(mu_cond, scale_cond)
            p_ref = tdist.Normal(mu_ref, scale_ref)

            xi = cond_dict["sample"].detach()
            #xi = ref_dict["sample"]

            loss_i = p_cond.log_prob(xi) - p_ref.log_prob(xi)
            loss += torch.sum(loss_i, dim=(1, 2, 3))

            # is weight could be different for a diffrent proposal
            lwi = loss_i
            ln_is_w += lwi

            if j == 0:
                break
        condition = condition.to("cuda")

        loss_0 = 1/2*torch.sum((Afwd(xi) - condition)**2, dim=1)    #likelihood.loss(xi, condition)
        print("loss0: ", loss_0)
        loss += loss_0 
        print("loss: ", loss)
        # we cannot use logsumexp
        # so hoping this deals with numerical issues
        # However this is biased?
        #weights = torch.exp(loss) 

        #if normed:  # this is biased 
        #    weights = torch.exp(lwi - lwi.max())
        #    weights = weights / weights.sum()
        
        #loss_final = (weights * loss).mean()
        #print("loss final: ", loss_final)
        print(loss.shape)
        loss_final = loss.var()
        return loss_final

    return loss




def get_loss_ddim(condition: torch.Tensor, pretrained_model, cond_model, sde, A, AT, normed=False):
    
    condition = condition

    def loss(condition=condition):

        xi = torch.randn((4,1,28,28)).to("cuda")
        device = "cuda"
        loss = 0.0
        ln_is_w = 0.0

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
        for i, j in tqdm(time_pairs):
            ones_vec = torch.ones(len(xi), device=device)

            if j<0: j=0

            time_step = (ones_vec * i, ones_vec * j)  # (t, tminus1)

            t = time_step[0]
            tminus1 = time_step[1]

            ref_dict = sde.ddim_sample(model, xi, t.long(), tminus1.long(), eta=0.8)

            def _cond_model(x, t, **kwargs):
                model_inp = torch.cat([x, ATy], dim=1)

                return model(x, t) + cond_model(model_inp, t)


            cond_dict = sde.ddim_sample(_cond_model, xi, t.long(), tminus1.long(), eta=0.8)

            mu_ref = ref_dict["mean"]
            log_var_ref = ref_dict["log_variance"]

            mu_cond = cond_dict["mean"]
            log_var_cond = cond_dict["log_variance"]
            
            scale_ref = (0.5 * log_var_ref).exp()
            scale_cond = (0.5 * log_var_cond).exp()

            p_cond = tdist.Normal(mu_cond, scale_cond)
            p_ref = tdist.Normal(mu_ref, scale_ref)

            xi = cond_dict["sample"]
            #xi = ref_dict["sample"]

            loss_i = p_cond.log_prob(xi) - p_ref.log_prob(xi)
            loss += torch.sum(loss_i, dim=(1, 2, 3))

            # is weight could be different for a diffrent proposal
            lwi = loss_i
            ln_is_w += lwi

            if j == 0:
                break
        condition = condition.to("cuda")

        loss_0 = 1/2*torch.sum((Afwd(xi) - condition)**2)    #likelihood.loss(xi, condition)
        print("loss0: ", loss_0)
        loss += loss_0 
        print("loss: ", loss)
        # we cannot use logsumexp
        # so hoping this deals with numerical issues
        # However this is biased?
        #weights = torch.exp(loss) 

        #if normed:  # this is biased 
        #    weights = torch.exp(lwi - lwi.max())
        #    weights = weights / weights.sum()
        
        #loss_final = (weights * loss).mean()
        #print("loss final: ", loss_final)
        loss_final = loss.mean()
        return loss_final

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

m = x.numel()/4

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
loss_fn = get_loss_ddim_v2(y_noise, normed=True, pretrained_model=model, cond_model=cond_model, sde=sde, A=Afwd, AT=Abwd)

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