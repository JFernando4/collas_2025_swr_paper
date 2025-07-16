import torch
from torch.optim import Optimizer
import numpy as np


@torch.no_grad()
def compute_utility(p: torch.Tensor, utility_name: str):
    if utility_name == "gradient":
        return torch.abs(p.grad * p).flatten()
    elif utility_name == "magnitude":
        return torch.abs(p).flatten()
    elif utility_name == "random":
        return torch.rand(size=(p.numel(), ), dtype=p.dtype, device=p.device)
    else:
        raise ValueError("Utility function not recognized")


@torch.no_grad()
def prune_weights(utility: torch.Tensor, pruning_method: str, reinit_factor: float) -> torch.Tensor:
    """
    Returns the indices of the weights to be pruned based on the utility and reinit_factor.
    """
    if pruning_method == "proportional":
        fraction_to_prune = utility.numel() * reinit_factor
        drop_num = int(fraction_to_prune) + np.random.binomial(n=1, p=fraction_to_prune % 1, size=None)
        if drop_num == 0: return torch.empty(0)
        indices = torch.argsort(utility)
        return indices[:drop_num]

    elif pruning_method == "threshold":
        prune_threshold = reinit_factor * utility.mean()
        return torch.where(utility <= prune_threshold)[0]

    else:
        raise ValueError("Pruning method not recognized.")


class SelectiveWeightReinitialization(Optimizer):

    def __init__(self,
                 params,
                 utility_function: str,
                 pruning_method: str,
                 param_means: list[float],
                 param_stds: list[float],
                 normal_reinit: list[bool],
                 reinit_freq: int = 0,
                 reinit_factor: float = 0.0,
                 decay_rate: float = 0.0
                 ):
        """

        Arguments:
            params: parameters of the network
            utility_function: str in ["gradient", "magnitude", "none"]
            pruning_method: str in ["proportional", "threshold"]
            param_means: list of floats with the mean value for each parameter at initialization
            param_stds: list of floats with the standard deviation or bound of uniform distribution for each parameter at initialization
            normal_reinit: list of bools indicating whether to reinitialize with normal or uniform distribution
            reinit_freq: float indicating how often to reinitialize
            reinit_factor: float used to determine the number of weights to reinitialize
            decay_rate: float used for computing the moving average o
        """
        defaults = dict(utility_function=utility_function, pruning_method=pruning_method,
                        means=param_means, stds=param_stds, normal_reinit=normal_reinit,
                        reinit_freq=reinit_freq, reinit_factor=reinit_factor, decay_rate=decay_rate)
        super().__init__(params, defaults)
        """
        Internal variables
        """
        self.reinit_indicator = False
        self.num_replaced = 0
        for group in self.param_groups:
            group.setdefault('current_step', 0)
            for p in group["params"]:
                self.state[p].setdefault("avg_utility", torch.zeros_like(p.flatten()))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            group["current_step"] += 1

            if group["decay_rate"] > 0.0:
                for p in group["params"]:
                    temp_utility = compute_utility(p, group["utility_function"])
                    self.state[p]["avg_utility"].mul_(group["decay_rate"]).add_(temp_utility, alpha=1.0-group["decay_rate"])

            if group["current_step"] % group["reinit_freq"] == 0:   # time to reinitialize
                for p, m, s, normal in zip(group["params"], group["means"], group["stds"], group["normal_reinit"]):
                    # compute utility
                    parameter_utility = compute_utility(p, group["utility_function"]) if group["decay_rate"] == 0.0 else self.state[p]["avg_utility"]
                    # get prune indices
                    reinit_indices = prune_weights(parameter_utility, group["pruning_method"], group["reinit_factor"])
                    num_reinit = reinit_indices.numel()
                    if num_reinit > 0:
                        # reinitialize
                        defaults = {"device": p.device, "dtype": p.dtype}
                        if normal:
                            new_values = torch.normal(mean=m, std=s, size=(num_reinit,), **defaults)
                        else:
                            new_values = torch.zeros(num_reinit, **defaults).uniform_(-s, s)

                        # Ensure in-place update on the original tensor
                        if not p.is_contiguous():
                            raise ValueError("Parameter tensor is not contiguous, which may lead to unexpected behavior.")
                        p.view(-1)[reinit_indices] = new_values

                        # reset utility and set flags
                        if group["decay_rate"] > 0.0:
                            self.state[p]["avg_utility"].zero_()
                        self.reinit_indicator = True
                        self.num_replaced += num_reinit

    def reset_reinit_indicator(self):
        self.reinit_indicator = False
        self.num_replaced = 0
