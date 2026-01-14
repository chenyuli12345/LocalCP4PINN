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
    def __init__(self, input_dim, hidden_dims, output_dim, mu_std, rho, prior_std=1.0, act_func=nn.Tanh()): #参数分别为输入维度、隐藏层维度列表、输出维度、mu的标准差、rho参数、先验标准差（这三个参数用于贝叶斯层）、激活函数
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
        # 输出层（也是贝叶斯层）
        layers.append(BayesianLinear(prev_dim, output_dim, mu_std, rho, prior_std))
        self.layers = nn.ModuleList(layers)  # not using Sequential because it's a mix of custom and activations

    def forward(self, x, sample: bool = True):
        out = x
        for layer in self.layers:
            # [贝叶斯层, 激活函数, 贝叶斯层, 激活函数, ..., 激活函数, 贝叶斯层]
            out = layer(out)  # 每一个都是贝叶斯层或者激活函数
        return out

    def kl_divergence(self):
        #加和每一个贝叶斯线形层的KL散度
        kl_total = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_total += layer.kl_divergence()
        return kl_total

#这段代码nll_gaussian计算的是高斯负对数似然（Negative Log-Likelihood, NLL）。在贝叶斯深度学习中不仅仅是让预测值逼近真实值（像MSE那样），而是从概率的角度看待问题：假设真实数据是由模型预测的均值加上某种高斯噪声产生的（也就是相当于假设观测数据y_true服从以模型预测y_pred为均值、以data_noise_guess为标准差的高斯分布），那么在这个假设下，观测到当前数据的概率有多大？要最大化这个概率（似然），等价于最小化负对数似然（NLL）。
#假设我们的观测数据𝑦_𝑡𝑟𝑢𝑒服从以模型预测值𝑦_𝑝𝑟𝑒𝑑为均值，以data_noise_guess(𝜎)为标准差的正态分布：𝑝(𝑦_𝑡𝑟𝑢𝑒∣𝑦_𝑝𝑟𝑒𝑑,𝜎)=1/(2𝜋𝜎)^(1/2)*exp⁡(−(𝑦_𝑡𝑟𝑢𝑒-𝑦_𝑝𝑟𝑒𝑑)^2/(2*𝜎^2)), 为了计算方便，我们对这个概率密度函数取对数：log⁡(1/(2𝜋𝜎^2)^(1/2))−(𝑦_𝑡𝑟𝑢𝑒-𝑦_𝑝𝑟𝑒𝑑)^2/(2*𝜎^2),其中第一项为常数项。因为在训练模型时是要“最小化损失”，而不是“最大化对数概率”，所以取负号，变成了 Negative Log-Likelihood (NLL)：(𝑦_𝑡𝑟𝑢𝑒-𝑦_𝑝𝑟𝑒𝑑)^2/(2*𝜎^2)-1/2*log⁡(1/(2𝜋𝜎^2))，其中第一项为数据拟合项，第二项为常数项（代码中被注释掉）
#data_noise_guess的物理意义，这个参数非常关键，它代表了我们对数据本身质量的预设（即偶然不确定性/Aleatoric Uncertainty）：Loss∝MSE/𝜎^2。如果data_noise_guess(σ)设得很小，这意味着你非常信任数据，认为数据非常精确，没什么噪声，结果是Loss会变得非常大，模型会受到强烈的惩罚，拼命去拟合每一个数据点（容易过拟合）。反之则认为数据很“脏”，含有很多噪声，结果是MSE被除以了一个大数，Loss变小了。模型会觉得“反正数据也不准，差不多对齐就行了”，此时模型更倾向于听从KL散度（先验）的指挥，保持简单的平滑曲线（容易欠拟合）。
#普通的MSE Loss只是单纯地衡量距离。NLL (Gaussian)是在衡量概率，它把误差项和数据的噪声方差联系在了一起。
    def nll_gaussian(self, y_pred, y_true, data_noise_guess=1.0): #参数项为预测值、真实值、数据噪声（默认1，设置越大越容易过拟合，设置越小越容易欠拟合）
        # omit constant
        mse = (y_pred - y_true).pow(2).sum()
        # const = N * torch.log(torch.tensor(2 * math.pi * data_noise_guess ** 2)) #常数项注释掉了
        nll = 0.5 * (mse / (data_noise_guess ** 2))
        return nll

#最后的贝叶斯神经网络的总损失是NLL和KL散度的和

