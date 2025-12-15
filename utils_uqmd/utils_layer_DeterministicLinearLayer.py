from utils_uqmd.interface_layer import BaseLayer
import torch
import torch.nn as nn
import math

#定义一个自定义的确定性线性层(DeterministicLinear)及其配套的初始化函数(xavier_init)。它的核心目的是创建一个行为上类似PyTorch原生nn.Linear的层，但在设计上是为了兼容贝叶斯神经网络（BNN）的混合架构训练框架。

#一个自定义的权重初始化函数，旨在模仿 PyTorch 默认的初始化行为，但在特定增益（gain）下进行调整
def xavier_init(weights, bias, gain=None):
    if gain is None:
        gain = nn.init.calculate_gain('tanh')  # ≈ 5/3
    nn.init.xavier_uniform_(weights, gain=gain)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


class DeterministicLinear(nn.Module):
    """一个标准的线性层（全连接层），但增加了一些接口以适配贝叶斯框架。"""
    def __init__(self, in_features, out_features, initialization=xavier_init):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Create learnable parameters: weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Weights initialization
        initialization(self.weight, self.bias)


    def forward(self, x): #前向传播函数
        return x.matmul(self.weight.t()) + self.bias

    def kl_divergence(self):
        """
        通过实现kl_divergence接口，它伪装成了一个“KL 散度为 0 的贝叶斯层”，从而可以无缝插入到贝叶斯神经网络的训练流程中，无需修改外部训练代码
        """
        return torch.tensor(0.0, device=self.weight.device)
