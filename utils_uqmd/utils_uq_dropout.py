import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils_uqmd.utils_model_pinn import *
from utils_uqmd.utils_layer_DeterministicLinearLayer import DeterministicLinear

#实现了一个带有蒙特卡洛 Dropout（Monte Carlo Dropout）进行不确定性量化的PINN

# ───────────────────────────────────────────────────────────────────────────────
#  1.  在每一个确定性层后插入了Drop-out的前馈神经网络
# ───────────────────────────────────────────────────────────────────────────────

class DeterministicDropoutNN(nn.Module):
    """
    与DeterministicFeedForwardNN相同的全连接网络，但每个隐藏层模块由Linear ▸ Dropout ▸ Activation组成。也就是在每一个全连接层后加入了Drop-out层。
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims,
                 output_dim: int,
                 p_drop: float, # dropout概率
                 act_func = nn.Tanh()
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.p_drop = p_drop
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                DeterministicLinear(prev, h),
                nn.Dropout(p_drop),           # <──新加的Dropout层
                act_func
            ])
            prev = h
        layers.append(DeterministicLinear(prev, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ───────────────────────────────────────────────────────────────────────────────
#  2.  带有标准物理损失并额外引入MC-Dropout不确定性量化的PINN封装器
# ───────────────────────────────────────────────────────────────────────────────

class DropoutPINN(nn.Module):
    """
    Physics-Informed Neural Network with Monte-Carlo Drop-out for UQ.

    Usage
    -----

    """
    #这一部份和标准PINN训练几乎完全一样
    def __init__(self,
                 pde_class,
                 input_dim: int,
                 hidden_dims,
                 output_dim: int,
                 p_drop: float = 0.1,
                 activation = nn.Tanh()):
        super().__init__()
        # Pure neural net part
        self.net = DeterministicDropoutNN(input_dim, hidden_dims,
                                          output_dim, p_drop, activation)
        self.pde = pde_class               # physics callbacks

    # Register backbone as a sub-module (it already is because of the attr)
    def forward(self, x):
        return self.net(x)

    def fit(self,
        # ------------ args ----------------
        coloc_pt_num,
        X_train, Y_train,
        # ----------- kwargs ---------------
        λ_pde=1.0, λ_ic=10.0, λ_bc=10.0, λ_data=5.0,
        epochs=20_000, lr=3e-3, print_every=500,
        scheduler_cls=StepLR, scheduler_kwargs={'step_size': 5000, 'gamma': 0.5},
        stop_schedule=40000
        ):
        
        device = X_train.device
        # move model to device
        self.to(device)
        # Optimizer
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # Scheduler
        scheduler = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None

        # Training History
        pde_loss_his = []
        bc_loss_his = []
        ic_loss_his = []
        data_loss_his = []

        self.train()
        for ep in range(1, epochs + 1):
            opt.zero_grad()

            # Init them as 0
            loss_data = 0
            loss_pde = 0
            loss_bc = 0
            loss_ic = 0

            # Data loss
            Y_pred = self.forward(X_train)
            loss_data = ((Y_pred - Y_train) ** 2).mean()
            loss = λ_data * loss_data
            data_loss_his.append(loss_data.item())

            # PDE residual
            if hasattr(self.pde, 'residual'):
                loss_pde = self.pde.residual(self, coloc_pt_num)
                loss += λ_pde * loss_pde
                pde_loss_his.append(loss_pde.item())

            # B.C. conditions
            if hasattr(self.pde, 'boundary_loss'):
                loss_bc = self.pde.boundary_loss(self)
                loss += λ_bc * loss_bc
                bc_loss_his.append(loss_bc.item())

            # I.C. conditions
            if hasattr(self.pde, 'ic_loss'):
                loss_ic = self.pde.ic_loss(self)
                loss += λ_ic * loss_ic
                ic_loss_his.append(loss_ic.item())
                
            loss.backward()
            opt.step()

            if ep <= stop_schedule:  # Stop decreasing the learning rate
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss.item())
                    elif isinstance(scheduler, StepLR):
                        scheduler.step()

            if (ep % print_every == 0 or ep == 1):  # Only start reporting after the warm-up Phase
                print(f"ep {ep:5d} | L={loss:.2e} | "
                      f"data={loss_data:.2e} | pde={loss_pde:.2e}  "
                      f"ic={loss_ic:.2e}  bc={loss_bc:.2e} | lr={opt.param_groups[0]['lr']:.2e}")



        return {"Data": data_loss_his, "Initial Condition Loss": ic_loss_his,
                "Boundary Condition Loss": bc_loss_his, "PDE Residue Loss": pde_loss_his}

    # -------------------------------------------------------------------------
    # 这一部份开始不一样，不确定性感知的预测，使用MC Dropout生成预测和相关的不确定性区间
    # -------------------------------------------------------------------------
    # Dropout模型
    @torch.inference_mode()
    def predict(
        # ------------ args ---------------
        self, alpha, #置信区间的显著性水平（例如对于95%的区间，alpha=0.05）
        X_test,  #需要进行预测和不确定性估计的输入数据
        # ----------- kwargs ---------------
        n_samples: int = 100, #在启用Dropout的情况下进行前向传播的次数。更多的样本通常能更好地估计不确定性。
        keep_dropout: bool = True #如果为True，则明确将Dropout层设置为训练模式 (self.train())，然后调用之后的enable_mc_dropout()方法 以确保Dropout在推理期间处于活动状态
    ):
        if keep_dropout:
            self.train()
            self.enable_mc_dropout()

        preds = []
        for _ in range(n_samples): #对所有测试数据X_test循环运行n_samples次正向传播。由于Dropout处于活动状态，每次传播都会产生略微不同的预测。
            preds.append(self.forward(X_test))
        preds = torch.stack(preds)                     #将这些预测堆叠起来，生成一个形状为(n_samples,num_test_points,output_dim)的张量。    # (S, N, out)
        mean = preds.mean(0) #获取多次预测的均值
        std  = preds.std(0) #获取多次预测的标准差

        # 计算置信区间（双侧(1−α)高斯置信区间）
        z = torch.tensor(
            abs(torch.distributions.Normal(0,1).icdf(torch.tensor(alpha/2))),
            device=preds.device, dtype=preds.dtype
        ) #计算与alpha水平对应的z分数，用于双侧高斯区间（例如对于alpha=0.05，z将近似为1.96）
        lower = mean - z*std
        upper = mean + z*std

        return (lower, upper) #返回 (lower, upper)，表示置信区间。

    # -------------------------------------------------------------------------
    # Helper: 在评估期间保持dropout层“开启”，这个辅助方法遍历PINN的所有子模块，并明确地将任何nn.Dropout层设置为train()模式
    # -------------------------------------------------------------------------
    def enable_mc_dropout(self):
        """
        Puts *all* Dropout sub-modules into training mode while leaving the rest
        of the network untouched.  Recommended before calling `predict_uq`.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


    @torch.inference_mode()
    def data_loss(self, X_test, Y_test):
        """Compute the data loss on the testing data set"""

        preds = self(X_test)
        loss  = torch.nn.functional.mse_loss(preds, Y_test,
                                             reduction="mean")
        # If the caller asked for a reduced value, return the Python float
        return loss.item() 