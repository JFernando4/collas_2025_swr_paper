"""
This is a modified version of torchvision's code for instantiating vision transformers. Here's a list of the changes
made to the source code:
    - Forward calls have a feature list argument to store the features of the network.
    - Replaced MLPBlock with CustomMLPBlock which works the same way but can also use continual backpropagation or
      ReDo on the feedforward layers.
To see the source code, go to: torchvision.models.vision_transformer (for torchvision==0.15.1)
"""

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once

from .reinitialization_layers import CBPLinear, ReDoLinear


class SequentialWithKeywordArguments(torch.nn.Sequential):

    """
    Sequential module that allows the use of keyword arguments in the forward pass
    """

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class CustomMLPBlock(torch.nn.Module):

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float,
                 replacement_rate: float = None, maturity_threshold: int = None,    # CBP parameters
                 reinit_frequency: int = None, reinit_threshold: float = None       # ReDo parameters
                 ) -> None:
        super().__init__()

        self.ff_1 = torch.nn.Linear(in_dim, mlp_dim, bias=True)
        self.act = torch.nn.GELU()
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.ff_2 = torch.nn.Linear(mlp_dim, in_dim, bias=True)
        self.dropout_2 = torch.nn.Dropout(dropout)

        self.cbp = None
        if (replacement_rate is not None) and (maturity_threshold is not None):
            self.cbp = CBPLinear(
                in_layer=self.ff_1,
                out_layer=self.ff_2,
                act_type="linear",
                replacement_rate=replacement_rate,
                init="kaiming",
                maturity_threshold=maturity_threshold
            )

        self.redo = None
        if (reinit_frequency is not None) and (reinit_threshold is not None):
            self.redo = ReDoLinear(
                in_layer=self.ff_1,
                out_layer=self.ff_2,
                act_type="linear",
                reinit_frequency=reinit_frequency,
                reinit_threshold=reinit_threshold,
                init="kaiming"
            )

        if (self.cbp is not None) and (self.redo is not None):
            raise ValueError("Cannot use both a CBP and a ReDo at the same time.")

    def forward(self, x: torch.Tensor, activations: list = None) -> torch.Tensor:

        x = self.ff_1(x)
        x = self.act(x)
        if self.cbp is not None:
            x = self.cbp(x)
        if self.redo is not None:
            x = self.redo(x)

        if activations is not None:
            activations.append(x)

        return self.dropout_2(self.ff_2(self.dropout_1(x)))


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        replacement_rate: float = None,
        maturity_threshold: int = None,
        reinit_frequency: int = None,
        reinit_threshold: float = None
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = CustomMLPBlock(hidden_dim, mlp_dim, dropout,
                                  replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, # cbp parameters
                                  reinit_frequency=reinit_frequency, reinit_threshold=reinit_threshold)     # redo parameters

    def forward(self, input: torch.Tensor, activations: list = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        if activations is not None:
            activations.append(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y, activations=activations)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        replacement_rate: float = None,
        maturity_threshold: int = None,
        reinit_frequency: int = None,
        reinit_threshold: float = None
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                replacement_rate=replacement_rate,
                maturity_threshold=maturity_threshold,
                reinit_frequency=reinit_frequency,
                reinit_threshold=reinit_threshold
            )
        self.layers = SequentialWithKeywordArguments(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, activations: list = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding

        return self.ln(self.layers(self.dropout(input), activations=activations))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        replacement_rate: float = None,
        maturity_threshold: int = None,
        reinit_frequency: int = None,
        reinit_threshold: float = None
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            reinit_frequency=reinit_frequency,
            reinit_threshold=reinit_threshold
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor, activations: list = None):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x, activations)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
