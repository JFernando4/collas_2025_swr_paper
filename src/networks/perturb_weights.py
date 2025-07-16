import torch
from math import sqrt


@torch.no_grad()
def perturb_weights(net: torch.nn.Module, parameter_noise_var: float):
    """ Injects noise to the parameters of the network

    Args:
        net (torch.nn.Module): The neural network whose parameters will be perturbed
        parameter_noise_var (float): The variance of the noise to be added to the parameters
    """
    for p in net.parameters():
        if p.requires_grad:  # only inject noise to learnable parameters
            noise = torch.randn_like(p) * sqrt(parameter_noise_var)
            p.add_(noise)
