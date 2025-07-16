import torch
from torch import nn
from math import sqrt


def call_reinit(m, i, o):
    m.reinit()


def log_features(m, i, o):
    with torch.no_grad():
        if m.decay_rate == 0:
            m.features = i[0]
        else:
            if m.features is None:
                m.features = (1 - m.decay_rate) * i[0]
            else:
                m.features = m.features * m.decay_rate + (1 - m.decay_rate) * i[0]


def get_layer_bound(layer, init, gain):
    if isinstance(layer, nn.Linear):
        if init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        else:   # kaiming
            bound = gain * sqrt(3 / layer.in_features)
        return bound
    else:
        raise ValueError("Only Linear layers are supported.")


def get_layer_std(layer, init, gain):
    if isinstance(layer, nn.Linear):
        if init == 'xavier':
            bound = gain * sqrt(2 / (layer.in_features + layer.out_features))
        else:   # kaiming
            bound = gain * sqrt(1 / layer.in_features)
        return bound
    else:
        raise ValueError("Only Linear layers are supported.")


class ReinitializationLinearLayer(nn.Module):

    def __init__(
            self,
            in_layer: nn.Module,
            out_layer: nn.Module,
            ln_layer: nn.LayerNorm = None,
            use_reparam_ln: bool = False,
            bn_layer: nn.BatchNorm1d = None,
            init: str = "kaiming",
            normal_init: bool = False,
            act_type: str = "relu",
            decay_rate: float = 0.0
    ):
        super().__init__()
        if type(in_layer) is not nn.Linear:
            raise Warning("Make sure in_layer is a linear layer")
        if type(out_layer) is not nn.Linear:
            raise Warning("Make sure out_layer is a linear layer")

        self.in_layer = in_layer
        self.out_layer = out_layer
        self.ln_layer = ln_layer
        self.bn_layer = bn_layer
        self.use_reparam_ln = use_reparam_ln
        self.normal_init = normal_init
        self.decay_rate = decay_rate
        self.ages = None
        self.features = None    # should be initialized by the user
        self.util_type = None   # should be initialized by the user
        self.util = None        # should be initialized by the user

        """
        Indicators for different events
        """
        self.replace_feature_event_indicator = False
        self.num_replaced = None

        """
        Calculate uniform distribution's bound or normal distribution std for random feature initialization
        """
        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))
        self.std = get_layer_std(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))

    def reset_indicators(self):
        self.replace_feature_event_indicator = False
        self.num_replaced = None

    def forward(self, _input):
        return _input

    def reinit(self):
        """
        Perform selective reinitialization
        """
        features_to_replace = self.get_features_to_reinit()
        self.reinit_features(features_to_replace)

    def reinit_features(self, features_to_replace: torch.Tensor):
        """
        Reset input and output weights for low utility features
        """
        with torch.no_grad():
            num_features_to_replace = features_to_replace.shape[0]
            if num_features_to_replace == 0: return
            self.in_layer.weight.data[features_to_replace, :] *= 0.0
            if self.normal_init:
                self.in_layer.weight.data[features_to_replace, :] += \
                    torch.empty(num_features_to_replace, self.in_layer.in_features, device=self.util.device).normal_(
                        mean=0, std=self.std)
            else:
                self.in_layer.weight.data[features_to_replace, :] += \
                    torch.empty(num_features_to_replace, self.in_layer.in_features, device=self.util.device).uniform_(
                        -self.bound, self.bound)
            self.in_layer.bias.data[features_to_replace] *= 0

            self.out_layer.weight.data[:, features_to_replace] = 0
            if self.ages is not None:
                self.ages[features_to_replace] = 0
            self.replace_feature_event_indicator = True
            self.num_replaced = num_features_to_replace

            """
            Reset the corresponding batchnorm/layernorm layers
            """
            if self.bn_layer is not None:
                self.bn_layer.bias.data[features_to_replace] = 0.0
                self.bn_layer.weight.data[features_to_replace] = 1.0
                self.bn_layer.running_mean.data[features_to_replace] = 0.0
                self.bn_layer.running_var.data[features_to_replace] = 1.0
            if self.ln_layer is not None:
                self.ln_layer.bias.data[features_to_replace] = 0.0
                new_ln_val = 0.0 if self.use_reparam_ln else 1.0
                self.ln_layer.weight.data[features_to_replace] = new_ln_val

    def compute_instantaneous_utility(self):
        if self.features is None:
            raise AttributeError("The features should be computed before calling this function! "
                                 "Something went wrong during the forward pass.")

        if self.util_type == "contribution":    # used in the CBP nature paper
            output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0)
            return output_weight_mag * self.features.abs().mean(dim=[i for i in range(self.features.ndim - 1)])
        elif self.util_type == "magnitude":     # most popular choice in the pruning literature
            return self.out_layer.weight.data.abs().mean(dim=0)
        elif self.util_type == "gradient":      # second most popular choice in the pruning literature
            weight_grad = self.out_layer.weight.grad
            weight_times_grad = weight_grad * self.out_layer.weight
            return weight_times_grad.sum(dim=0).abs()
        elif self.util_type == "activation":    # used in the ReDo paper
            return self.features.abs().mean(dim=[i for i in range(self.features.ndim - 1)])
        else:
            raise ValueError(f"Unknown utility type: {self.util_type}")

    def get_features_to_reinit(self) -> torch.Tensor:
        raise NotImplementedError

class CBPLinear(ReinitializationLinearLayer):
    def __init__(
            self,
            in_layer: nn.Module,
            out_layer: nn.Module,
            ln_layer: nn.LayerNorm = None,
            use_reparam_ln: bool = False,
            bn_layer: nn.BatchNorm1d = None,
            init: str = 'kaiming',
            normal_init: bool = False,
            act_type: str = 'relu',
            decay_rate: float = 0.0,
            replacement_rate: float = 0.0,
            maturity_threshold: int = 1000,
            util_type: str = 'contribution',
            **kwargs
    ):
        super().__init__(in_layer, out_layer, ln_layer, use_reparam_ln, bn_layer, init, normal_init, act_type, decay_rate)
        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        """
        Register hooks
        """
        if self.replacement_rate > 0.0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)
        """
        Utility of all features/neurons
        """
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)

    def get_features_to_reinit(self):
        """
        Returns: Features to replace
        """
        features_to_replace = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1
        """
        Calculate number of features to replace
        """
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:  return features_to_replace

        num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
        self.accumulated_num_features_to_replace += num_new_features_to_replace
        if self.accumulated_num_features_to_replace < 1:    return features_to_replace

        num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
        self.accumulated_num_features_to_replace -= num_new_features_to_replace
        """
        Calculate feature utility
        """
        self.util.data = self.compute_instantaneous_utility()
        """
        Find features with smallest utility
        """
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace = new_features_to_replace
        return features_to_replace


class ReDoLinear(ReinitializationLinearLayer):

    def __init__(
            self,
            in_layer: nn.Module,
            out_layer: nn.Module,
            ln_layer: nn.LayerNorm = None,
            use_reparam_ln: bool = False,
            bn_layer: nn.BatchNorm1d = None,
            init='kaiming',
            normal_init: bool = False,
            act_type='relu',
            decay_rate: float = 0.0,
            reinit_frequency=0,
            reinit_threshold=0.1,
            util_type='activation',
            **kwargs
    ):
        super().__init__(in_layer, out_layer, ln_layer, use_reparam_ln, bn_layer, init, normal_init, act_type, decay_rate)
        """
        Define the hyper-parameters of the algorithm
        """
        self.reinit_frequency = reinit_frequency
        self.reinit_threshold = reinit_threshold
        self.util_type = util_type
        """
        Register hooks
        """
        if self.reinit_threshold > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)
        """
        Utility of all features/neurons
        """
        self.step_count = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)

    def get_features_to_reinit(self):
        """
        Returns: Features to replace
        """
        features_to_replace = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.step_count += 1
        """
        Determine whether to replace features
        """
        if self.step_count % self.reinit_frequency != 0: return features_to_replace
        """
        Calculate feature utility
        """
        self.util.data = self.compute_instantaneous_utility()
        """
        Get average utility and determine threshold
        """
        average_utility = self.util.data.mean()
        threshold = average_utility * self.reinit_threshold
        """
        Find features below the threshold
        """
        return torch.where(self.util.data <= threshold)[0]
