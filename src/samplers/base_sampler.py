''' 
Inspired to https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py 
'''
from typing import Optional, Any, Dict, Tuple

import os
import torchvision
import numpy as np
import torch

from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class BaseSampler:
    def __init__(self, 
        score, 
        sde,
        sampl_fn: callable,
        sample_kwargs: Dict,
        device: Optional[Any] = None
        ) -> None:

        self.score = score
        self.sde = sde
        self.sampl_fn = sampl_fn
        self.sample_kwargs = sample_kwargs
        self.device = device
    
    def sample(self,
        logg_kwargs: Dict = {},
        logging: bool = True 
        ) -> Tensor:

        
        num_steps = self.sample_kwargs['num_steps']
        
        time_steps = np.linspace(1., self.sample_kwargs['eps'], self.sample_kwargs['num_steps'])
        __iter__ = time_steps
        

        step_size = time_steps[0] - time_steps[1]

        init_x = self.sde.prior_sampling([self.sample_kwargs['batch_size'], *self.sample_kwargs['im_shape']]).to(self.device)

        x = init_x
        i = 0
        pbar = tqdm(__iter__)
        for step in pbar:
            ones_vec = torch.ones(self.sample_kwargs['batch_size'], device=self.device)
            time_step = ones_vec * step #
  
            x, x_mean = self.sampl_fn(
                score=self.score,
                sde=self.sde,
                x=x,
                time_step=time_step,
                step_size=step_size,
                )

        return x_mean 