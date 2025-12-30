from utils_uqmd.interface_layer import BaseLayer
import torch
import torch.nn as nn
import math

#贝叶斯线性层的实现，贝叶斯的权重和偏置服从概率分布（通常是高斯分布）。训练过程中不直接学习权重值，而是学习权重分布的参数（均值μ和标准差σ）

# def default_mu_rho(in_features, out_features,
#                    mu_std=0.1, rho=-3.0,
#                    mu_mean=0.0,
#                    prior_std=1.0):

#     # Weights and Biases Distribution Initialization
#     weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(mu_mean, mu_std))
#     weight_rho = nn.Parameter(torch.empty(out_features, in_features).fill_(rho))
#     bias_mu = nn.Parameter(torch.empty(out_features).normal_(mu_mean, mu_std))
#     bias_rho = nn.Parameter(torch.empty(out_features).fill_(rho))

#     # Std of the prior distribution
#     prior_std = prior_std

#     return weight_mu, weight_rho, bias_mu, bias_rho, prior_std


import torch
import torch.nn as nn

#初始化函数，用于初始化贝叶斯层中需要学习的参数，包括权重和偏置的均值（mu）和rho（用于计算标准差）
def default_mu_rho(in_features, out_features,
                  mu_std=0.1,        # 兼容旧API，实际代码中未被使用
                  rho=-3.0,         #决定初始标准差的参数
                  prior_std=1.0,   #先验分布的标准差
                  gain=None):
    if gain is None:
        #如果后续使用 tanh 激活函数，推荐使用 5/3 的增益，保持方差稳定
        gain = nn.init.calculate_gain('tanh')  # 5/3

    #权重的均值
    weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    nn.init.xavier_uniform_(weight_mu, gain=gain)

    #偏置的均值
    bias_mu = nn.Parameter(torch.zeros(out_features))

    #Rho参数，用于计算权重和偏置的标准差，同时通过softplus函数确保标准差为正值
    weight_rho = nn.Parameter(torch.full((out_features, in_features), rho))
    bias_rho   = nn.Parameter(torch.full((out_features,), rho))

    return weight_mu, weight_rho, bias_mu, bias_rho, float(prior_std)

#贝叶斯层的权重w和偏执不是一个固定的数，而是一个概率分布（通常假设为高斯分布w∼N(μ,σ^2)，其中μ是均值，σ是标准差）
class BayesianLinearLayer(BaseLayer): #继承自BaseLayer
    """
    全因子化的贝叶斯线性层，在调用时将sample=False，即可得到后验分布的均值（确定性输出）。
    """

    def __init__(self, in_features, out_features,
                 mu_std=0.1, rho=-3.0, prior_std=1.0,
                 initialization=default_mu_rho): #参数分别为输入维度、输出维度、mu的标准差(好像没用，初始化权重和偏置的均值μ时直接用0)、rho参数（初始时刻的rho值，然后通过softplus函数即变为最终权重和偏置的标准差）、先验标准差（初始状态下模型有多不确定，即预设的权重和偏置的高斯分布的标准差，均值为0，预设的标准差越小，相当于强正则化，强迫模型学到的权重必须非常接近0，且分布非常窄；预设的标准差越大，相当于弱正则化，允许模型学到的权重这就偏离0较远，或者拥有较大的不确定性）
        super().__init__()
        (self.weight_mu, self.weight_rho,
         self.bias_mu,  self.bias_rho,
         self.prior_std) = initialization(in_features, out_features,
                                          mu_std=mu_std, rho=rho,
                                          prior_std=prior_std) #调用初始化函数获得四个可学习的参数
        self.log2pi = math.log(2 * math.pi) #一个常数

    # ---------- 辅助函数 ----------
    @staticmethod
    def _softplus(x):
        #Softplus函数：平滑的ReLU，将(-inf,inf)映射到(0,inf)
        return torch.log1p(torch.exp(x))

    # ---------- forward ----------
    def forward(self, x, *, sample: bool = True):
        """
        sample=True: 训练时使用。从分布中随机采样权重，模拟不确定性。
        sample=False: 预测/测试时使用。直接使用均值，相当于普通的确定性神经网络。
        """
        if sample: #训练时
            w_sigma = self._softplus(self.weight_rho)
            b_sigma = self._softplus(self.bias_rho)
            eps_w   = torch.randn_like(w_sigma)
            eps_b   = torch.randn_like(b_sigma)
            weight  = self.weight_mu + w_sigma * eps_w
            bias    = self.bias_mu  + b_sigma * eps_b
        else:                       # posterior mean
            weight, bias = self.weight_mu, self.bias_mu
        return x.matmul(weight.t()) + bias #执行线性变换: y = xW^T + b

    # ---------- 当前层KL散度的计算 ----------
    #希望后验分布（学到的）不要偏离先验分布（初始印象）太远，因此在训练过程中会计算KL散度作为正则化项（防止过拟合）。     
    #代码中假设权重的“先验分布”和“后验分布”都是高斯分布（正态分布）。其中后验分布q(w)（模型学到的），均值为μ，标准差为σ。先验分布p(w)（我们预设的）：均值为0，标准差为σ_p（代码中的 prior_std）。对于这两个单变量高斯分布，它们之间的 KL 散度公式是：D_KL(q||p) = ln(σ_p/σ) + (σ^2 + (μ - 0)^2) / (2σ_p^2) - 1/2，这个公式由三部分组成。
    #prior_std：先验信念（正则化强度）。含义是在训练前人为设定的、理想中的权重分布的标准差。作用是定义了KL散度中的目标分布p(w)：p(w)=N(0,prior_std^2)。在训练中计算KL散度，目的是让学到的后验分布q(w)靠近这个先验分布p(w)。较小的 prior_std（如0.1）相当于强正则化（类似于很强的L2 Weight Decay），强迫模型学到的权重必须非常接近0，且分布非常窄。较大的prior_std（如1.0）相当于弱正则化，允许模型学到的权重这就偏离0较远，或者拥有较大的不确定性.   
    def kl_divergence(self):
        w_sigma = self._softplus(self.weight_rho) #计算后验的标准差
        b_sigma = self._softplus(self.bias_rho)
        prior_var = self.prior_std ** 2 #先验分布的方差

        #这里sum的作用是因为一个线性层有成千上万个权重（W）和偏置（b），我们假设每个权重之间是相互独立的，所以把所有权重的 KL 散度累加起来，得到这一层总的“代价”
        kl_w = torch.sum(torch.log(self.prior_std / w_sigma) + #ln(σ_p/σ)（对比两个分布的“胖瘦”（宽度）。如果模型学到的σ非常小（分布非常窄/瘦），这一项的值就会变得很大。这相当于在惩罚那些过于“自信”的神经元。）
                         0.5 * (w_sigma ** 2 + self.weight_mu ** 2) / prior_var #(σ^2 + μ^2) / (2σ_p^2)（对比两个分布的“位置”。如果均值μ离0很远，或者σ很大，这一项就会增加。这相当于L2正则化（权重衰减），它防止权重数值炸裂）
                         - 0.5 #常数项，保证当两个分布完全一样时，KL散度为0
                         )
        kl_b = torch.sum(torch.log(self.prior_std / b_sigma) +
                         0.5 * (b_sigma ** 2 + self.bias_mu ** 2) / prior_var - 0.5)
        return kl_w + kl_b #返回该层的总KL散度


# Explain::
# class BayesianLinearLayer(BaseLayer):
#     """ 一层 Bayesian Linear Layer
#         Bayesian Linear layer with Gaussian weight and bias priors and variational posteriors.
#     """

#     def __init__(self, in_features, out_features, mu_std, rho, prior_std, initialization=default_mu_rho):
#         super().__init__()
#         # Mean and log-variance (or rho) for weights and biases as learnable parameters
#         self.in_features = in_features
#         self.out_features = out_features

#         # ------------------------------ Model's Parameters ------------------------------------------
#         # Initialize means (mu) to small random values, and rho to a small negative (so sigma ~ small)
#         # Since σ must be strictly positive, so we optimize rho, compute σ by softplus(rho)
#         # So, we are still learning the std σ, but indirectly
#         (self.weight_mu, self.weight_rho, self.bias_mu, self.bias_rho,
#          self.prior_std) = initialization(in_features, out_features, mu_std=mu_std, rho=rho, prior_std=prior_std)

#         # Prior standard deviation (fixed)
#         self.log2pi = math.log(2 * math.pi)  # for potential use in exact logprob if needed

#     def forward(self, x):
#         # Sample the std  σ  of the weights and biases (the reparameterization trick)
#         weoight_sigma = torh.log1p(torch.exp(self.weight_rho))  # softplus to ensure positivity
#         bias_sigma = torch.log1p(torch.exp(self.bias_rho))

#         # Sample ε ∼ 𝒩(0,1) for weights and baises
#         eps_w = torch.randn_like(weight_sigma)
#         eps_b = torch.randn_like(bias_sigma)

#         # Sample from 𝒩(mu, sigma^2) through variable transformation
#         # 这样, 我们就能 update `mu` 和 `sigma`(rho)
#         weight = self.weight_mu + weight_sigma * eps_w
#         bias = self.bias_mu + bias_sigma * eps_b

#         # Linear layer computation xWᵀ + b
#         return x.matmul(weight.t()) + bias  # the utput of this bayesian linear layer

#     def kl_divergence(self):
#         # Compute KL divergence KL[q(w,b) || p(w,b)] for this layer (sum over all weights and biases)
#         # Assuming factorized Gaussian posteriors and Gaussian priors N(0, prior_std^2):contentReference[oaicite:2]{index=2}.
#         weight_sigma = torch.log1p(torch.exp(self.weight_rho))
#         bias_sigma = torch.log1p(torch.exp(self.bias_rho))
#         # KL for each weight: log(prior_sigma/post_sigma) + (post_sigma^2 + mu^2)/(2*prior_sigma^2) - 0.5
#         prior_var = self.prior_std ** 2

#         # Compute the KL value using the formula for Gaussian
#         #   = log(prior_std/posterior_std) + 0.5 * (posterior_std**2 + posterior_mu**2) / [prior_std^2] - 1)
#         # For numerical stability, avoid log of 0 by using weight_sigma (softplus ensures >0)
#         kl_weight = torch.sum(torch.log(self.prior_std / weight_sigma) +
#                               0.5 * (weight_sigma ** 2 + self.weight_mu ** 2) / prior_var - 0.5)
#         kl_bias = torch.sum(torch.log(self.prior_std / bias_sigma) +
#                             0.5 * (bias_sigma ** 2 + self.bias_mu ** 2) / prior_var - 0.5)
#         return kl_weight + kl_bias