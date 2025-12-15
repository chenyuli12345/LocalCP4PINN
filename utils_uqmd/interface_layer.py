from abc import ABC, abstractmethod #来自Python标准库abc。这表明 BaseLayer 是一个抽象类，不能直接实例化，必须被子类继承。它的主要作用是定义接口规范
import torch
import torch.nn as nn

class BaseLayer(ABC, nn.Module):
    """
    #定义了一个名为BaseLayer的抽象基类，它是为PINN设计的，旨在兼容确定性(Deterministic)和贝叶斯(Bayesian)两种类型的层。这种设计是为了创建一个通用的接口，使得构建混合模型（既包含普通层也包含贝叶斯层）变得容易。
    所有继承的层必须重写forward和kl_divergence方法。
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        这强制要求任何继承 BaseLayer 的非抽象子类必须实现 forward 方法。如果子类没有定义 forward，在实例化时会报错。
        """
        pass

    def kl_divergence(self) -> torch.Tensor:
        """
        返回这一层的KL散度。计算的是后验分布和先验分布之间的差距。
        在贝叶斯神经网络中，我们学习权重的分布而不是具体的值。KL散度用于衡量“预测的权重分布”与“先验分布”之间的差异。贝叶斯层的损失函数通常包含两部分：Loss = Data_Fitting_Loss + KL_Divergence。
        默认返回0.0：对于确定性层（如标准的 nn.Linear），权重是固定的数值，没有分布的概念（或者说分布是狄拉克δ函数），因此不存在KL散度损失。这个默认实现允许我们在模型中混合使用确定性层和贝叶斯层，而不需要为确定性层特殊编写代码——它们只会给总损失增加 0。
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)
