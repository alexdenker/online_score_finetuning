"""
File from Francisco Vargas.

"""

import yaml 
import numpy as np 
import os 
import torch 


from model.diffusion import Diffusion

def get_loss_is(condition: torch.Tensor, normed=False):
    """
    condition: [1, C, H, W]
    """
    assert condition.ndim == 4
    assert condition.shape[0] == 1

    def loss():

        xi = torch.randn_like(condition)
        device = condition.device
        loss = 0.0
        ln_is_w = 0.0

        # network_ref = network_cond.unconditional
        tdist = torch.distributions        

        for time in reversed(range(ddpm.Ns)):
            batched_i = torch.ones((len(condition))).fill_(time).long().to(device)


            x0_cond = conditioning.x0_model(xi, batched_i, ddpm, "cond")
            mu_cond, _, log_var_cond, _ = ddpm.p_mean_variance(x_start=x0_cond, x=xi, i=batched_i)
            scale_cond = (0.5 * log_var_cond).exp()
            x0_ref = conditioning.x0_model(xi, batched_i, ddpm, "ref")
            mu_ref, _, log_var_ref, _ = ddpm.p_mean_variance(x_start=x0_ref, x=xi, i=batched_i)
            scale_ref = (0.5 * log_var_ref).exp()

            p_cond = tdist.Normal(mu_cond, scale_cond)
            p_ref = tdist.Normal(mu_ref, scale_ref)

            xi, _ = ddpm.p_sample(x0_ref, xi, batched_i)

            loss_i = p_cond.log_prob(xi) - p_ref.log_prob(xi)
            loss += torch.sum(loss_i, dim=(1, 2, 3))
        
            # is weight could be different for a diffrent proposal
            lwi = loss_i
            ln_is_w += lwi
        
        loss_0 = likelihood.loss(xi, condition)
        loss += loss_0 

        # we cannot use logsumexp
        # so hoping this deals with numerical issues
        # However this is biased?
        weights = torch.exp(loss)

        if normed:  # this is biased 
            weights = torch.exp(lwi - lwi.max())
            weights = weights / weights.sum()
        
        loss_final = (weights * loss).mean()

        return loss_final
    return loss

base_path = "/localdata/AlexanderDenker/score_based_baseline/MNIST"

with open(os.path.join(base_path, "report.yaml"), "r") as f:
    cfg_dict = yaml.safe_load(f)

print(cfg_dict)


"""

losses = []
optimizer = torch.optim.Adam(conditioning.parameters(), lr=1e-3)
loss_fn = get_loss_is(condition[:1], normed=False)
for i in range(100):
    optimizer.zero_grad()
    loss_value = loss_fn()
    loss_value.backward()
    optimizer.step()

    losses.append(loss_value)
"""