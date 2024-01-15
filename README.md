# Online Finetuning Objective 

This is a simple implementation of the online finetuning objective using **torchsde** as a backbone. 

I created a new repository because:

1. Different (smaller) model architecture, which is trained using the score matching loss instead of the epsilon matching loss. 
2. Only the continuous formulation of score-based generative modeling, so different sampler (Euler-Maruyama) etc. 

## Torchsde 

The SDE solvers in torchsde can only integrate forward in time. So, I needed to reverse the time direction in our reverse SDE. 

The solver takes a vector of time steps as an input (**ts**). But apparently the step size is controlled by an additional input (**dt**), and is not inferred by **ts**.

## Examples 

1. test_torchsde.py: Simple Euler-Maruyama solver to draw samples from a pre-trained MNIST model.