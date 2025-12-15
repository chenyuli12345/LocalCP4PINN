from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BasePINNModel(ABC, nn.Module):
    """
    定义了一个名为BasePINNModel的抽象基类，旨在为PINN提供一个统一的架构，同时支持确定性和贝叶斯两种建模方式。
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        一个装饰器，强制要求任何继承此类的子类必须实现forward方法。如果不实现，实例化子类时会报错。
        """
        pass

    def kl_divergence(self) -> torch.Tensor:
        """
        对于确定性模型：不存在参数分布的概念，因此KL散度为0。这个基类方法默认返回0，使得普通PINN可以直接继承而无需修改此方法。
        对于贝叶斯模型：子类需要重写此方法，返回实际计算出的KL散度值。
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def mc_predict(self, x: torch.Tensor, n_samples: int = 100) -> tuple:
        """
        MC dropout方法，一种通过多次采样来估计模型输出分布的方法。
        代码逻辑：运行self(x) (即 forward pass) n_samples次。detach()表示从计算图中分离，因为预测阶段通常不需要反向传播，节省显存。torch.stack(preds)则将多次预测结果堆叠成一个新的张量。最后返回均值 (mean) 和方差 (var)。
        针对不同模型的行为：确定性模型每次forward结果完全一样。因此，方差将全为0，均值就是单次预测结果。贝叶斯模型每次forward时，权重会从分布中重新采样。Dropout模型每次forward时，会随机丢弃部分神经元。因此，每次输出略有不同。通过计算这100次的方差，可以量化模型对该输入的不确定性。
        """
        preds = [self(x).detach() for _ in range(n_samples)]
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.var(dim=0)
