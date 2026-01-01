# ────────────────────────────────────────────────────────────────────────────────
#  DistanceUQPINN  –  PINN + distance-based uncertainty in one class
# ────────────────────────────────────────────────────────────────────────────────
import torch, math, numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils_uqmd.interface_model import BasePINNModel
from utils_uqmd.utils_layer_DeterministicLinearLayer import DeterministicLinear

#构建基于距离度量不确定性的PINN。这个模型本身是确定性的，通过计算“测试点距离训练数据的远近”来估算不确定性。


class DistanceUQPINN(BasePINNModel):
    """
    一种物理信息神经网络，其预测带完全由输入（特征）空间或隐（潜）空间中的 k-近邻（k-NN）距离决定。该方法未使用CP校准。
    """

    # ───────────────────── constructor ─────────────────────
    def __init__(
        self,
        pde_class,
        input_dim:   int,
        hidden_dims: list | int,
        output_dim:  int,
        activation:  nn.Module = nn.Tanh(),
        device:      torch.device = "cpu"
    ):
        super().__init__()
        self.pde = pde_class
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Build feed-forward backbone
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [DeterministicLinear(prev, h), activation]
            prev = h
        layers += [DeterministicLinear(prev, output_dim)]
        self.layers = nn.ModuleList(layers)
        self.device = device
        self.to(self.device)

        # 占位符：用于缓存训练数据，以便后续计算距离
        self._X_train_cached = None


    # ───────────────────── forward pass ─────────────────────
    def forward(self, x: torch.Tensor, *, return_hidden: bool = False):
        """
        Parameters
        ----------
        x : (N, input_dim)形状的张量
        return_hidden : 如果是Ture，同时返回最后一个隐藏层的输出
        """
        out, hidden = x, None
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, DeterministicLinear):
                hidden = out          # 捕获线性变换后、激活函数前的数值
        return (out, hidden) if return_hidden else out #如果return_hidden为True，则返回最后一层的输出和最后一层未经激活函数的输出（大部分时候直接等于输出）


    # ───────────────────── trainer ─────────────────────
    def fit(
        self,
        coloc_pt_num:  int,
        X_train:       torch.Tensor,
        Y_train:       torch.Tensor,
        *,
        λ_pde:         float          = 1.0,
        λ_ic:          float          = 10.0,
        λ_bc:          float          = 10.0,
        λ_data:        float          = 5.0,
        epochs:        int            = 20_000,
        lr:            float          = 3e-3,
        print_every:   int            = 500,
        scheduler_cls                 = StepLR,
        scheduler_kwargs: dict        = {'step_size': 5000, 'gamma': 0.7},
        stop_schedule: int            = 40_000
    ):
        """Standard PINN training (MSE + PDE/BC/IC losses)."""
        #缓存训练数据（这里是带输出的数据点，即观测数据点，不是配位点），这是为了后续预测时计算距离
        self._X_train_cached = X_train.detach()

        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None

        for ep in range(1, epochs + 1):
            opt.zero_grad()
            loss = λ_data * ((self.forward(X_train) - Y_train) ** 2).mean()

            if hasattr(self.pde, 'residual'): #检查self.pde这个对象里，是否存在一个叫做 'residual' 的属性或方法
                loss += λ_pde * self.pde.residual(self, coloc_pt_num)
            if hasattr(self.pde, 'boundary_loss'):
                loss += λ_bc * self.pde.boundary_loss(self)
            if hasattr(self.pde, 'ic_loss'):
                loss += λ_ic * self.pde.ic_loss(self)

            loss.backward(); opt.step()

            #日志与学习率调度(Scheduler):PINN的Loss Landscape通常非常崎岖（非凸、有很多局部极小值）。刚开始使用较大的学习率（lr的初始值）快速下降。随着训练进行，衰减学习率（gamma=0.7）以便微调，进入更深的谷底。并且为了防止学习率变得过小导致训练停滞，代码加了一个保险：在超过stop_schedule步后，即使还在训练，也不再降低学习率了。
            if ep % print_every == 0 or ep == 1:
                print(f"ep {ep:5d} | L={loss:.2e} | lr={opt.param_groups[0]['lr']:.1e}")

            if ep <= stop_schedule and sched:
                if isinstance(sched, ReduceLROnPlateau):
                    sched.step(loss.item())
                else:
                    sched.step()


    # ───────────────────── distance helpers ─────────────────────
    @staticmethod #装饰器，表明是一个静态方法，不需要实例对象(self参数)即可调用
    def _to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x) #一个静态工具方法，将张量转换为NumPy数组

    #两种距离度量方式
    def _feature_dist(self, X_test: torch.Tensor, k: int): #特征空间距离
        """在原始输入空间中计算k近邻距离，在原始输入X空间计算测试点与训练集最近的k个点的平均距离。"""
        trn, tst = self._to_np(self._X_train_cached), self._to_np(X_test) #获取缓存的训练点/观测数据训练点的坐标 (trn) 和新增的测试点坐标 (tst)
        nnm = NearestNeighbors(n_neighbors=k).fit(trn) #这里的fit实际上是在构建一种数据结构（通常是KD-Tree或Ball-Tree），参数k是k临近，fit中的参数是缓存的训练点/观测数据训练点的坐标
        d, _ = nnm.kneighbors(tst) #参数是新增的测试点坐标，d是一个形状为 (N_test, k) 的矩阵。每一行对应一个测试点，每一行里有k 个数值，代表离它最近的第 1、第 2...第k个训练点的欧几里得距离，第二个_ (Indices)则是对应邻居的索引，_把这个返回值忽略掉
        return d.mean(axis=1)  # (N,)，对每一行的k个距离求平均，得到了一个长度为N_test的向量，代表每个新增的测试点所最近的k个训练点的平均欧几里得距离

    def _latent_dist(self, X_test: torch.Tensor, k: int):
        """最后一个隐藏层的平均k-NN距离（实际上用的是输出层的，原因在forward方法上）"""
        with torch.no_grad(): #禁用梯度
            H_trn = self.forward(self._X_train_cached.to(self.device), return_hidden=True)[1] #获取缓存的训练点/观测数据训练点的隐藏特征
            H_tst = self.forward(X_test.to(self.device),               return_hidden=True)[1] #获取新增的测试点的隐藏特征
        trn, tst = self._to_np(H_trn), self._to_np(H_tst)
        nnm = NearestNeighbors(n_neighbors=k).fit(trn)
        d, _ = nnm.kneighbors(tst)
        return d.mean(axis=1)


    # ───────────────────── 预测和不确定性量化 ─────────────────────
    #将训练好的网络权重（用于预测值）和计算出的空间距离（用于预测带）结合起来，输出一个带置信区间的预测结果
    def predict(
        self,
        alpha:       float,
        X_test:      torch.Tensor,
        *,
        heuristic_u: str   = "feature",   # 'feature' | 'latent'
        n_samples:           int   = 20,
        return_band: bool  = True,
    ):
        """
        输入参数
        ----------
        X_test : (N, input_dim)形状的tensor，新增的测试点的坐标
        heuristic_u : 'feature' | 'latent'，选择距离度量方式:'feature'或'latent'
        k : int              k-NN距离的邻居数量
        scale : float        将几何距离转化为预测的标准差（不确定度） → 预测的 σ̂
        return_band : bool   如果是False，只返回点的预测值
        alpha : float        误覆盖率，如0.05代表 95% 置信区间

        返回值
        -------
        • ŷ                        如果return_band是False
        • (lower, upper) tensors    如果return_band是True
        """
        X_test = X_test.to(self.device)

        # ─── 点的预测 ───────────────────────────────────────────
        with torch.no_grad(): #不计算梯度
            y_pred = self.forward(X_test)          # (N, out_dim)形状，获得新增的测试点的输出

        if not return_band:
            return y_pred                          # 只返回均值

        if self._X_train_cached is None:
            raise RuntimeError("Must call fit() before predict() so training data is cached.")

        # ─── 确定不确定性的大小σ̂ ───────────────────────
        dist = (self._feature_dist(X_test, n_samples) if heuristic_u == "feature"
                else self._latent_dist(X_test, n_samples))                     # (N,)，计算新增测试点在“原始空间”或“隐层空间”里离训练集的平均距离dist
        sigma_hat = torch.from_numpy(dist)[:, None].to(self.device)   # (N,1)，将距离(dist)等同于标准差 (σ^)。即如果一个点离训练集1.0单位远，就假设预测误差的标准差也是1.0

        #计算分位数
        alpha_tensor = torch.as_tensor(1.0 - alpha / 2.0, device=self.device) #用户允许的“出错概率”（误覆盖率），alpha=0.05意味着想要95%的置信区间。
        z = torch.distributions.Normal(0.0, 1.0).icdf(alpha_tensor)

        # ─── (1-α) interval:  ŷ ± z·σ̂ ───────────────────────────────
        lower = y_pred - z * sigma_hat
        upper = y_pred + z * sigma_hat
        return (lower, upper) #返回上下界


    @torch.inference_mode() #PyTorch较新版本中no_grad()的增强版，性能更好，专门用于推理。
    def data_loss(self, X_test, Y_test):
        """计算测试数据集上的均方误差测试损失"""
        preds = self(X_test) #执行前向传播,等同于调用self.forward(X_test)
        loss  = torch.nn.functional.mse_loss(preds, Y_test,
                                             reduction="mean") #计算均方误差 (MSE)：(1/N) * Σ(预测值 - 真实值)^2
        # If the caller asked for a reduced value, return the Python float
        return loss.item()  #返回python原生数值
