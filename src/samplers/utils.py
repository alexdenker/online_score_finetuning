from typing import Optional, Any, Dict, Tuple, Union

import torch
import numpy as np
import torch.nn as nn

from torch import Tensor

def Euler_Maruyama_sde_predictor(
    score,
    sde,
    x: Tensor,
    time_step: Tensor,
    step_size: float,
    ) -> Tuple[Tensor, Tensor]:
    
    s = score(x, time_step)

    drift, diffusion = sde.sde(x, time_step)

    x_mean = x - (drift - diffusion[:, None, None, None].pow(2)*s)*step_size
    noise = torch.sqrt(diffusion[:, None, None, None].pow(2)*step_size)*torch.randn_like(x)

    x = x_mean + noise 

    return x.detach(), x_mean.detach()
