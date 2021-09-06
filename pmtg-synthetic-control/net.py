"""Policy Network for Synthetic Control Problem

Author: Ren Liu
"""
import math
from typing import Any, List

from torch import nn
import torch


def init_weights(m: Any):
    """initialize the network weights"""
    if type(m) == nn.Linear:
        stdv = 1.0 / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


class VanillaPMTG(nn.Module):
    """PMTG policy network using simple MLP architecture"""

    def __init__(self, in_dims: int = 3, out_dims: int = 4, arch: List[int] = [32]) -> None:
        """initialization

        Args:
            in_dims (int, optional): input dimensions (x, y, current_time). Defaults to 3.
            out_dims (int, optional): output dimentions (a_x, a_y, u_fb_x, u_fb_y). Defaults to 4.
            arch (list, optional): number of hidden dims. Defaults to [32].
        """
        super(VanillaPMTG, self).__init__()

        modules = [nn.Linear(in_dims, arch[0]), nn.ReLU()]
        for i in range(1, len(arch) - 1):
            modules.append(nn.Linear(arch[i - 1], arch[i]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(arch[-1], out_dims))
        self.linear_relu_stack = nn.Sequential(*modules)

        self.linear_relu_stack.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output tg parameters & actions from feedback network (u_fb)"""
        x = self.linear_relu_stack(x)

        return x
