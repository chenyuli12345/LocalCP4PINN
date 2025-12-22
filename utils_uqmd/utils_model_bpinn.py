from utils_uqmd.interface_model import BasePINNModel
from utils_uqmd.utils_layer_BayesianLinearLayer import BayesianLinearLayer as BayesianLinear

import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
##########################################################
# TODO: 将PINN的网络结构转换为贝叶斯网络结构
##########################################################


class BayesianFeedForwardNN(BasePINNModel):
    """具有贝叶斯线性层的前馈神经网络 (for VI)."""
    def __init__(self, input_dim, hidden_dims, output_dim, mu_std, rho, prior_std=1.0, act_func=nn.Tanh()): #参数分别为输入维度、隐藏层维度列表、输出维度、mu的标准差、rho参数、先验标准差、激活函数
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers = []
        prev_dim = input_dim
        # 遍历 hidden_dims（隐藏层维度列表），在每一层都放置一个 BayesianLinear
        for h in hidden_dims:
            layers.append(BayesianLinear(prev_dim, h, mu_std, rho, prior_std))  # in_feat, out_feat, prior_std
            layers.append(act_func)
            prev_dim = h
        # Final output layer (Bayesian linear as well)
        layers.append(BayesianLinear(prev_dim, output_dim, mu_std, rho, prior_std))
        self.layers = nn.ModuleList(layers)  # not using Sequential because it's a mix of custom and activations

    def forward(self, x, sample: bool = True):
        out = x
        for layer in self.layers:
            # [BL, act, BL, act, ..., act, BL] go through all the layers
            out = layer(out)  # BayesianLinear or activation
        return out

    def kl_divergence(self):
        # Sum KL from all BayesianLinear layers
        kl_total = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_total += layer.kl_divergence()
        return kl_total

    def nll_gaussian(self, y_pred, y_true, data_noise_guess=1.0):
        # omit constant
        mse = (y_pred - y_true).pow(2).sum()
        # const = N * torch.log(torch.tensor(2 * math.pi * data_noise_guess ** 2))
        nll = 0.5 * (mse / (data_noise_guess ** 2))
        return nll



