from abc import ABC, abstractmethod
import torch

class BaseUQPINN(ABC):
    """
    为在PINN中进行不确定性量化提供一套统一的编程规范和接口
    """

    def __init__(self, model: torch.nn.Module, pde, x_collocation: torch.Tensor,
                 x_boundary: torch.Tensor = None, boundary_values=None):
        """
        初始化方法：
        Parameters:
            model:底层的神经网络。它可以是一个普通的确定性网络（用于Ensemble或Dropout方法），也可以是一个专门的贝叶斯网络层（用于BNN方法）
            pde: 一个封装了物理方程的对象。应该包含.residual(model, x)和.boundary_loss(model)方法。这使得这个UQ框架是通用的，不绑定于特定的物理问题（如热传导、流体力学），只要pde对象符合接口即可。
            x_collocation: 用于计算残差的配位点
            x_boundary: 边界条件和初始条件的点
            boundary_values: 真实的边界/初始条件值
        """
        self.model = model
        self.pde = pde
        self.x_collocation = x_collocation
        self.x_boundary = x_boundary
        self.boundary_values = boundary_values

    @abstractmethod
    def train(self, num_epochs: int = 10000, lr: float = 1e-3, print_every: int = 500):
        """
        抽象方法，任何继承此类的子类必须实现具体的训练循环。不同的UQ方法训练方式完全不同：
        标准/Dropout PINN：通常最小化 MSE_Res + MSE_BC。
        贝叶斯 PINN (VI)：最小化 KL散度 - Log似然 (ELBO)。
        HMC (Hamiltonian Monte Carlo)：使用采样而非梯度下降。
        这个接口允许上述所有方法共存，只需改变 train 的内部实现。
        """
        pass

    @abstractmethod
    def predict(self, x_test: torch.Tensor, **kwargs):
        """
        这个方法规定了所有UQ模型的预测必须返回两个值：
        mean (均值)：模型对物理量的最佳预测。
        uncertainty (不确定性)：模型对该预测的置信度（通常是标准差、方差或置信区间宽度）。
        代码**kwargs: 提供了灵活性。例如，对于 MC Dropout 或 BNN，你可能需要传入 n_samples=100 来指定采样的次数。
        """
        pass
