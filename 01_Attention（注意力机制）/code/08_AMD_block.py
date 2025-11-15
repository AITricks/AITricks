"""
即插即用模块集合 - AMD架构的核心组件
这些模块可以独立使用或组合使用，用于时间序列预测任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """
    可逆实例归一化模块 (Reversible Instance Normalization)
    用于时间序列的归一化和反归一化，提高模型的泛化能力
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: 特征或通道数
        :param eps: 数值稳定性参数
        :param affine: 是否使用可学习的仿射参数
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class MDM(nn.Module):
    """
    多尺度可分解混合模块 (Multi-Scale Decomposable Mixing)
    将输入分解为多个尺度并进行混合，捕获不同时间尺度的信息
    """
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        """
        :param input_shape: 输入形状 [seq_len, feature_num]
        :param k: 多尺度层数
        :param c: 尺度缩放因子
        :param layernorm: 是否使用层归一化
        """
        super(MDM, self).__init__()
        self.seq_len = input_shape[0]
        self.k = k
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        """
        :param x: [batch_size, feature_num, seq_len]
        :return: [batch_size, feature_num, seq_len]
        """
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x
        # x [batch_size, feature_num, seq_len]
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        # [batch_size, feature_num, seq_len]
        return sample_x[n - 1]


class DDI(nn.Module):
    """
    双依赖交互模块 (Dual Dependency Interaction)
    建模不同尺度之间的动态交互关系
    """
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True):
        """
        :param input_shape: 输入形状 [seq_len, feature_num]
        :param dropout: dropout率
        :param patch: patch大小
        :param alpha: 特征交互权重
        :param layernorm: 是否使用层归一化
        """
        super(DDI, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.input_shape = input_shape
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.n_history = 1
        self.alpha = alpha
        self.patch = patch

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [batch_size, feature_num, seq_len]
        :return: [batch_size, feature_num, seq_len]
        """
        # [batch_size, feature_num, seq_len]
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # input [batch_size, feature_num, self.n_history * patch]
            input = output[:, :, i - self.n_history * self.patch: i]
            # input [batch_size, feature_num, self.n_history * patch]
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)
            # aggregation
            # [batch_size, feature_num, patch]
            input = F.gelu(self.agg(input))  # self.n_history * patch -> patch
            input = self.dropout_t(input)
            # input [batch_size, feature_num, patch]
            # input = torch.squeeze(input, dim=-1)
            tmp = input + x[:, :, i: i + self.patch]

            res = tmp

            # [batch_size, feature_num, patch]
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                # [batch_size, patch, feature_num]
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)
            output[:, :, i: i + self.patch] = res + self.alpha * tmp

        # [batch_size, feature_num, seq_len]
        return output


class TopKGating(nn.Module):
    """
    Top-K门控机制
    用于选择最重要的专家进行预测
    """
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        """
        :param input_dim: 输入维度
        :param num_experts: 专家数量
        :param top_k: 选择的top-k专家数
        :param noise_epsilon: 噪声epsilon
        """
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

    def decompostion_tp(self, x, alpha=10):
        """
        Top-K分解函数
        :param x: [batch_size, num_experts]
        :param alpha: 分解参数
        :return: [batch_size, num_experts]
        """
        # x [batch_size, seq_len]
        output = torch.zeros_like(x)
        # [batch_size]
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        # [batch_size, num_expert]
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        # [batch_size, seq_len]
        return output

    def forward(self, x):
        """
        :param x: [batch_size, seq_len]
        :return: [batch_size, num_experts] 门控权重
        """
        # [batch_size, seq_len]

        x = self.gate(x)
        clean_logits = x
        # [batch_size, num_experts]

        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.decompostion_tp(logits)
        gates = self.softmax(logits)

        return gates


class Expert(nn.Module):
    """
    专家网络
    单个预测器，用于处理特定的时间模式
    """
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param hidden_dim: 隐藏层维度
        :param dropout: dropout率
        """
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        :param x: [batch_size, input_dim]
        :return: [batch_size, output_dim]
        """
        return self.net(x)


class AMS(nn.Module):
    """
    自适应多预测器合成模块 (Adaptive Multi-predictor Synthesis)
    根据时间模式自适应选择并组合多个预测器
    """
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, num_experts=4, top_k=2):
        """
        :param input_shape: 输入形状 [seq_len, feature_num]
        :param pred_len: 预测长度
        :param ff_dim: 前馈网络维度
        :param dropout: dropout率
        :param loss_coef: 损失系数
        :param num_experts: 专家数量
        :param top_k: top-k专家数
        """
        super(AMS, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len

        self.gating = TopKGating(input_shape[0], num_experts, top_k)

        self.experts = nn.ModuleList(
            [Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)])
        self.loss_coef = loss_coef
        assert (self.top_k <= self.num_experts)

    def cv_squared(self, x):
        """
        计算变异系数的平方，用于负载均衡损失
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        """
        :param x: [batch_size, feature_num, seq_len]
        :param time_embedding: [batch_size, feature_num, seq_len] 时间嵌入
        :return: output [batch_size, feature_num, pred_len], loss 负载均衡损失
        """
        # [batch_size, feature_num, seq_len]
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        # [feature_num, batch_size, seq_len]
        x = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)

        output = torch.zeros(feature_num, batch_size, self.pred_len).to(x.device)
        loss = 0

        for i in range(feature_num):
            input = x[i]
            time_info = time_embedding[i]
            # x[i]  [batch_size, seq_len]
            gates = self.gating(time_info)

            # expert_outputs [batch_size, num_experts, pred_len]
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len).to(x.device)

            for j in range(self.num_experts):
                expert_outputs[j, :, :] = self.experts[j](input)
            expert_outputs = torch.transpose(expert_outputs, 0, 1)
            # gates [batch_size, num_experts, pred_len]
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)
            # batch_output [batch_size, pred_len]
            batch_output = (gates * expert_outputs).sum(1)
            output[i, :, :] = batch_output

            importance = gates.sum(0)
            loss += self.loss_coef * self.cv_squared(importance)

        # [feature_num, batch_size, seq_len]
        output = torch.transpose(output, 0, 1)
        # [batch_size, feature_num, seq_len]

        return output, loss


def test_modules():
    """
    测试所有即插即用模块的功能
    """
    print("=" * 60)
    print("开始测试即插即用模块...")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    # 测试参数
    batch_size = 4
    seq_len = 96
    feature_num = 7
    pred_len = 24
    
    # 1. 测试 RevIN
    print("1. 测试 RevIN 模块")
    print("-" * 60)
    revin = RevIN(num_features=feature_num).to(device)
    x_revin = torch.randn(batch_size, seq_len, feature_num).to(device)
    x_norm = revin(x_revin, mode='norm')
    x_denorm = revin(x_norm, mode='denorm', target_slice=slice(None))
    print(f"输入形状: {x_revin.shape}")
    print(f"归一化后形状: {x_norm.shape}")
    print(f"反归一化后形状: {x_denorm.shape}")
    print(f"RevIN 测试通过 ✓\n")
    
    # 2. 测试 MDM
    print("2. 测试 MDM 模块")
    print("-" * 60)
    mdm = MDM(input_shape=(seq_len, feature_num), k=3, c=2, layernorm=True).to(device)
    x_mdm = torch.randn(batch_size, feature_num, seq_len).to(device)
    x_mdm_out = mdm(x_mdm)
    print(f"输入形状: {x_mdm.shape}")
    print(f"输出形状: {x_mdm_out.shape}")
    print(f"MDM 测试通过 ✓\n")
    
    # 3. 测试 DDI
    print("3. 测试 DDI 模块")
    print("-" * 60)
    ddi = DDI(input_shape=(seq_len, feature_num), dropout=0.1, patch=12, alpha=0.5, layernorm=True).to(device)
    x_ddi = torch.randn(batch_size, feature_num, seq_len).to(device)
    x_ddi_out = ddi(x_ddi)
    print(f"输入形状: {x_ddi.shape}")
    print(f"输出形状: {x_ddi_out.shape}")
    print(f"DDI 测试通过 ✓\n")
    
    # 4. 测试 TopKGating
    print("4. 测试 TopKGating 模块")
    print("-" * 60)
    topk_gating = TopKGating(input_dim=seq_len, num_experts=4, top_k=2).to(device)
    x_gating = torch.randn(batch_size, seq_len).to(device)
    gates = topk_gating(x_gating)
    print(f"输入形状: {x_gating.shape}")
    print(f"门控权重形状: {gates.shape}")
    print(f"门控权重和: {gates.sum(dim=1)}")  # 应该接近1.0
    print(f"TopKGating 测试通过 ✓\n")
    
    # 5. 测试 Expert
    print("5. 测试 Expert 模块")
    print("-" * 60)
    expert = Expert(input_dim=seq_len, output_dim=pred_len, hidden_dim=512, dropout=0.1).to(device)
    x_expert = torch.randn(batch_size, seq_len).to(device)
    x_expert_out = expert(x_expert)
    print(f"输入形状: {x_expert.shape}")
    print(f"输出形状: {x_expert_out.shape}")
    print(f"Expert 测试通过 ✓\n")
    
    # 6. 测试 AMS
    print("6. 测试 AMS 模块")
    print("-" * 60)
    ams = AMS(input_shape=(seq_len, feature_num), pred_len=pred_len, 
              ff_dim=512, dropout=0.1, num_experts=4, top_k=2).to(device)
    x_ams = torch.randn(batch_size, feature_num, seq_len).to(device)
    time_emb = torch.randn(batch_size, feature_num, seq_len).to(device)
    x_ams_out, moe_loss = ams(x_ams, time_emb)
    print(f"输入形状: {x_ams.shape}")
    print(f"时间嵌入形状: {time_emb.shape}")
    print(f"输出形状: {x_ams_out.shape}")
    print(f"MoE损失: {moe_loss.item():.6f}")
    print(f"AMS 测试通过 ✓\n")
    
    # 7. 测试模块组合
    print("7. 测试模块组合 (完整流程)")
    print("-" * 60)
    # 模拟完整的前向传播流程
    x_combined = torch.randn(batch_size, seq_len, feature_num).to(device)
    
    # RevIN归一化
    x_combined = revin(x_combined, mode='norm')
    
    # 转置为 [batch, feature, seq]
    x_combined = x_combined.transpose(1, 2)
    
    # MDM处理
    time_embedding = mdm(x_combined)
    
    # DDI处理
    x_combined = ddi(x_combined)
    
    # AMS预测
    pred, loss = ams(x_combined, time_embedding)
    
    # 转回 [batch, pred_len, feature]
    pred = pred.transpose(1, 2)
    
    # RevIN反归一化
    pred = revin(pred, mode='denorm', target_slice=slice(None))
    
    print(f"组合输入形状: {x_combined.shape}")
    print(f"组合输出形状: {pred.shape}")
    print(f"组合MoE损失: {loss.item():.6f}")
    print(f"模块组合测试通过 ✓\n")
    
    print("=" * 60)
    print("所有模块测试完成！✓")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    # 运行测试
    test_modules()

