import torch
import math

from ..networks.permuted_mnist_network import ThreeHiddenLayerNetwork
from ..networks.torchvision_modified_vit import VisionTransformer


def get_network_init_parameters(net: torch.nn.Module, reinit_strat: str, **kwargs) -> tuple:
    assert reinit_strat in ["resample", "mean"]

    reparam_ln = kwargs["reparam_ln"] if "reparam_ln" in kwargs.keys() else False
    if isinstance(net, ThreeHiddenLayerNetwork):
        return get_permuted_mnist_network_init_parameters(net, reinit_strat, reparam_ln)
    if isinstance(net, VisionTransformer):
        return get_vit_network_init_parameters(net, reinit_strat, reparam_ln)
    else:
        raise ValueError("Network not recognized.")


def get_permuted_mnist_network_init_parameters(net: ThreeHiddenLayerNetwork, reinit_strat: str, reparam_ln: bool) -> tuple:
    """
    returns lists with means, stds, and normal_reinit indicator with an entry for each parameter in the network
    """

    means, stds, normal_reinit = [], [], []

    for n, p in net.named_parameters():
        is_weight = "weight" in n
        is_bias = "bias" in n
        is_layer_norm = "ln" in n

        temp_mean = 0.0
        temp_std = 0.0
        if is_weight and is_layer_norm:                 # layer norm weights
            temp_mean = 0.0 if reparam_ln else 1.0
        elif is_bias:                                   # bias terms always initialized to zero
            pass
        else:                                           # weight of matrices
            gain = torch.nn.init.calculate_gain("relu")
            kaiming_normal_std = gain / (torch.nn.init._calculate_correct_fan(p, "fan_in") ** 0.5)
            temp_std = kaiming_normal_std if reinit_strat == "resample" else 0.0

        means.append(temp_mean); stds.append(temp_std); normal_reinit.append(True)

    return means, stds, normal_reinit


def get_vit_network_init_parameters(net: VisionTransformer, reinit_strat: str, reparam_ln: bool) -> tuple:
    """
    returns lists with means, stds, and normal_reinit indicator with an entry for each parameter in the network
    """
    is_mean_reinit = reinit_strat == "mean"
    means, stds_or_bounds, normal_reinit = [], [], []
    mlp_fan = 0

    for n, p in net.named_parameters():
        is_weight = "weight" in n
        is_bias = "bias" in n
        is_layer_norm = (".ln_1." in n) or (".ln_2." in n) or (".ln." in n)

        temp_mean, temp_std_or_bound, temp_normal_reinit = 0.0, 0.0, False
        if n == "class_token":                          # init = 0.0
            pass
        elif "pos_embedding" in n:                      # init = (0.0, 0.02)
            temp_std_or_bound = 0.0 if is_mean_reinit else 0.02
            temp_normal_reinit = True
        elif is_weight and is_layer_norm:               # Layer norm scaling factor, init = 0.0 if reparam else 1.0
            temp_mean = 0.0 if reparam_ln else 1.0
            temp_normal_reinit = True   # so that mean is 1.0 in case of regular ln, uniform doesn't use mean
        elif is_bias:
            if ".mlp." in n:                            # init = Uniform(-1/sqrt(weight_fan_in), 1/sqrt(weight_fan_in))
                bound = 1 / math.sqrt(mlp_fan) if mlp_fan > 0.0 else 0.0    # the fan-in of the corresponding weight matrix
                temp_std_or_bound = 0.0 if is_mean_reinit else bound
            else:                                       # init = 0.0
                pass
        else:
            if "heads" in n:                            # heads = output layer, init = 0.0
                pass
            elif ("conv_proj" in n):                    # init = Normal(0.0, 1/sqrt(fan_in))
                # technically, the init function is truncated normal, but the standard deviation is so small that the
                # chance of the truncation kicking in is essentially 0 for practical purposes
                fan_in = torch.nn.init._calculate_correct_fan(p, "fan_in")
                temp_std_or_bound = 0.0 if is_mean_reinit else math.sqrt(1/fan_in)
                temp_normal_reinit = True
            elif ("out_proj" in n) or (".mlp." in n):   # init = Kaiming Uniform init with a = math.sqrt(5)
                # the initialization reduces to Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
                fan_in = torch.nn.init._calculate_correct_fan(p, "fan_in")
                temp_std_or_bound = 0.0 if is_mean_reinit else 1/math.sqrt(fan_in) # Calculate uniform bounds from standard deviation
                if ".mlp." in n:
                    mlp_fan = fan_in
            else:                                      # init = Xavier uniform init
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(p)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                temp_std_or_bound = 0.0 if is_mean_reinit else math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        means.append(temp_mean); stds_or_bounds.append(temp_std_or_bound); normal_reinit.append(temp_normal_reinit)
    return means, stds_or_bounds, normal_reinit
