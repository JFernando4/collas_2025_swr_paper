from .reinitialization_layers import CBPLinear, ReDoLinear
from .reparameterized_layer_norm import ReparameterizedLayerNorm

from math import sqrt
import torch

INPUT_DIMS = 784
OUTPUT_DIMS = 10


class ThreeHiddenLayerNetwork(torch.nn.Module):

    def __init__(self,
                 hidden_dim: int = 10,
                 activation_function: str = "relu",
                 use_skip_connections: bool = False,
                 preactivation_skip_connection: bool = False,
                 use_cbp=False,
                 maturity_threshold: int = None,
                 replacement_rate: float = None,
                 cbp_utility: str = "contribution",
                 use_redo=False,
                 reinit_frequency: int = None,
                 reinit_threshold: float = None,
                 redo_utility: str = "original",
                 use_layer_norm: bool = False,
                 preactivation_layer_norm: bool = False,
                 reinit_after_ln: bool = False,
                 use_reparam_ln: bool = False,
                 use_crelu: bool = False):
        """
        Three-layer ReLU network with continual backpropagation for MNIST
        args:
            hidden_dim: (int) number of hidden units in each layer
            activation_function: (str) activation function to use, one of "relu", "sigmoid", "tanh", "leaky_relu", "gelu", "silu"
            use_skip_connections: (bool) whether to use skip connections
            preactivation_skip_connection: (bool) whether to add skip connections before or after activation
            use_layer_norm (bool): whether to use layer normalization
            preactivation_layer_norm (bool): whether to apply layer normalization before or after activation
            reinit_after_ln: (bool) whether to store features for redo or cbp before or after layer norm, only matters if layer norm is used
            use_reparam_ln (bool): whether to use reparameterized layer normalization
            use_crelu (bool): whether to use crelu
        """
        super().__init__()
        # crelu options
        self.use_crelu = use_crelu
        input_dim_scaling = 1
        if self.use_crelu:
            assert hidden_dim % 2 == 0
            hidden_dim = hidden_dim // 2
            input_dim_scaling = 2
        # skip connections options
        self.use_skip_connections = use_skip_connections
        self.preactivation_skip_connection = preactivation_skip_connection
        # layer norm
        self.use_layer_norm = use_layer_norm
        self.preactivation_layer_norm = preactivation_layer_norm
        self.reinit_after_ln = reinit_after_ln
        ln_class = ReparameterizedLayerNorm if use_reparam_ln else torch.nn.LayerNorm
        # reinitialization layer
        self.use_reinit = (use_redo or use_cbp)
        if use_redo and use_cbp: raise ValueError("Cannot use ReDo and CBP at the same time!")
        reinit_class = CBPLinear if use_cbp else ReDoLinear
        reinit_layer_arguments = {"replacement_rate": replacement_rate, "maturity_threshold": maturity_threshold,
                                  "reinit_frequency": reinit_frequency, "reinit_threshold": reinit_threshold,
                                  "util_type": cbp_utility if use_cbp else redo_utility, "normal_init": True}
        act_func = {"relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid, "tanh": torch.nn.Tanh, "leaky_relu": torch.nn.LeakyReLU,
                    "gelu": torch.nn.GELU, "silu": torch.nn.SiLU}[activation_function]
        if activation_function != "relu" and self.use_crelu: raise ValueError("Activation function must be 'relu' when using crelu!")
        # first layer
        self.ff_1 = torch.nn.Linear(INPUT_DIMS, out_features=hidden_dim, bias=True)
        self.act_1 = act_func()
        self.neg_act_1 = act_func()
        self.ln_1 = ln_class(hidden_dim * input_dim_scaling) if self.use_layer_norm else None
        self.weights_per_feature_1 = INPUT_DIMS + hidden_dim        # cbp and redo would reinit this many weights
        # second layer
        second_layer_dim = hidden_dim
        self.ff_2 = torch.nn.Linear(hidden_dim * input_dim_scaling, out_features=second_layer_dim, bias=True)
        self.reinit_layer_1 = reinit_class(in_layer=self.ff_1, out_layer=self.ff_2, ln_layer=self.ln_1, **reinit_layer_arguments) if self.use_reinit else None
        self.act_2 = act_func()
        self.neg_act_2 = act_func()
        self.ln_2 = ln_class(second_layer_dim * input_dim_scaling) if self.use_layer_norm else None
        self.weights_per_feature_2 = hidden_dim * 2
        # third layer
        self.ff_3 = torch.nn.Linear(second_layer_dim * input_dim_scaling, out_features=hidden_dim, bias=True)
        self.reinit_layer_2 = reinit_class(in_layer=self.ff_2, out_layer=self.ff_3, ln_layer=self.ln_2, **reinit_layer_arguments) if self.use_reinit else None
        self.act_3 = act_func()
        self.neg_act_3 = act_func()
        self.ln_3 = ln_class(hidden_dim * input_dim_scaling) if self.use_layer_norm else None
        self.weights_per_feature_3 = hidden_dim + OUTPUT_DIMS
        # output layer
        self.out = torch.nn.Linear(hidden_dim * input_dim_scaling, OUTPUT_DIMS, bias=True)
        self.reinit_layer_3 = reinit_class(in_layer=self.ff_3, out_layer=self.out, ln_layer=self.ln_3, **reinit_layer_arguments) if self.use_reinit else None

    def forward(self, x: torch.Tensor, activations: list = None) -> torch.Tensor:
        # first hidden layer
        x = self.ff_1(x)
        if self.use_layer_norm and self.preactivation_layer_norm:           # use layer norm before activations
            x = self.ln_1(x)
        x = torch.cat([self.act_1(x), self.neg_act_1(-x)], dim=-1) if self.use_crelu else self.act_1(x)
        if activations is not None: activations.append(x)                   # store activations
        res = x                                                             # store residual connection
        if self.reinit_layer_1 is not None and not self.reinit_after_ln:    # log features using cbp or redo before layer norm
            x = self.reinit_layer_1(x)
        if self.use_layer_norm and not self.preactivation_layer_norm:       # use layer norm after activation
            x = self.ln_1(x)
        if self.reinit_layer_1 is not None and self.reinit_after_ln:        # log features using cbp or redo after layer norm
            x = self.reinit_layer_1(x)

        # second hidden layer
        x = self.ff_2(x)
        if self.use_layer_norm and self.preactivation_layer_norm:
            x = self.ln_2(x)
        if self.use_skip_connections and self.preactivation_skip_connection:    # add residual connection before activation
            x = x + res
        x = torch.cat([self.act_2(x), self.neg_act_2(-x)], dim=-1) if self.use_crelu else self.act_2(x)
        if activations is not None: activations.append(x)
        if self.use_skip_connections and not self.preactivation_skip_connection:    # add residual connection after activation
            x = x + res
        res = x
        if self.reinit_layer_2 is not None and not self.reinit_after_ln:
            x = self.reinit_layer_2(x)
        if self.use_layer_norm and not self.preactivation_layer_norm:
            x = self.ln_2(x)
        if self.reinit_layer_2 is not None and self.reinit_after_ln:
            x = self.reinit_layer_2(x)

        # third hidden layer
        x = self.ff_3(x)
        if self.use_layer_norm and self.preactivation_layer_norm:
            x = self.ln_3(x)
        if self.use_skip_connections and self.preactivation_skip_connection:
            x = x + res
        x = torch.cat([self.act_3(x), self.neg_act_3(-x)], dim=-1) if self.use_crelu else self.act_3(x)
        if activations is not None: activations.append(x)
        if self.use_skip_connections and not self.preactivation_skip_connection:
            x = x + res
        if self.reinit_layer_3 is not None and not self.reinit_after_ln:
            x = self.reinit_layer_3(x)
        if self.use_layer_norm and not self.preactivation_layer_norm:
            x = self.ln_3(x)
        if self.reinit_layer_3 is not None and self.reinit_after_ln:
            x = self.reinit_layer_3(x)

        return self.out(x)

    def feature_replace_event_indicator(self):
        if not self.use_reinit: return False

        return (self.reinit_layer_1.replace_feature_event_indicator or
                self.reinit_layer_2.replace_feature_event_indicator or
                self.reinit_layer_3.replace_feature_event_indicator)

    def reset_indicators(self):
        if not self.use_reinit: return

        self.reinit_layer_1.reset_indicators()
        self.reinit_layer_2.reset_indicators()
        self.reinit_layer_3.reset_indicators()

    def num_replaced(self):
        if not self.use_reinit: return (0, 0, 0)
        nr_1 = 0.0 if not self.reinit_layer_1.replace_feature_event_indicator else self.reinit_layer_1.num_replaced
        nr_2 = 0.0 if not self.reinit_layer_2.replace_feature_event_indicator else self.reinit_layer_2.num_replaced
        nr_3 = 0.0 if not self.reinit_layer_3.replace_feature_event_indicator else self.reinit_layer_3.num_replaced
        return nr_1 * self.weights_per_feature_1, nr_2 * self.weights_per_feature_2, nr_3 * self.weights_per_feature_3

def init_three_hidden_layer_network_weights(m, nonlinearity='relu'):
    """
    Initializes weights using kaiming initialization
    :param m: torch format for network layer
    :param nonlinearity: (str) specifies the type of activation used in layer m
    :return: None, operation is done in-place
    """
    from torchvision.models import VisionTransformer
    if isinstance(m, torch.nn.Linear):
        if nonlinearity in ["gelu", "silu"]:
            torch.nn.init.xavier_normal_(m.weight)
        else:
            negative_slope = 1e-2 if nonlinearity == 'leaky_relu' else 0.0
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity, a=negative_slope)
        if m.bias is not None:
            m.bias.data.fill_(0.0)