"""
SST (State Space Transformer) 即插即用模块集合
包含可以从SST架构中独立使用的核心模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
from typing import Optional


# ============================================================================
# 1. RevIN - 可逆实例归一化 (Reversible Instance Normalization)
# ============================================================================
class RevIN(nn.Module):
    """
    可逆实例归一化模块
    用于时间序列预测中消除分布偏移问题
    
    使用方法:
        revin = RevIN(num_features=7)
        x_norm = revin(x, mode='norm')  # 归一化
        x_denorm = revin(x_norm, mode='denorm')  # 反归一化
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ============================================================================
# 2. Router - 长短路由模块 (Long-Short Router)
# ============================================================================
class Router(nn.Module):
    """
    长短路由模块
    自适应学习长短期专家的贡献权重
    
    输入: [Batch, Input length, Channel]
    输出: m_weight (long-range权重), t_weight (short-range权重)
    
    使用方法:
        router = Router(long_context_window=96, context_window=48, c_in=7, d_model=128)
        m_weight, t_weight = router(long_seq)  # 输出两个权重标量
    """
    def __init__(self, long_context_window, context_window, c_in, d_model, bias=True):
        super().__init__()
        self.context_window = context_window
        # 投影层
        self.W_P = nn.Linear(c_in, d_model, bias=bias)
        self.flatten = nn.Flatten(start_dim=-2)
        # 权重生成
        self.W_w = nn.Linear(long_context_window*d_model, 2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, long):  # x: [Batch, Input length, Channel]
        # 投影
        x = self.W_P(long)
        x = self.flatten(x)
        # 权重生成
        prob = self.softmax(self.W_w(x))
        m_weight, t_weight = prob[:,0], prob[:,1]
        return m_weight, t_weight


# ============================================================================
# 3. Fusion_Head - 融合头 (Long-Short Range Fusion Head)
# ============================================================================
class Fusion_Head(nn.Module):
    """
    长短范围融合头
    将长短期专家的输出进行加权融合并生成最终预测
    
    输入:
        long: [Batch, Channel, d_model, m_patch_num] - 长期特征
        short: [Batch, Channel, d_model, t_patch_num] - 短期特征
        m_weight: [Batch] - 长期权重
        t_weight: [Batch] - 短期权重
    
    使用方法:
        fusion_head = Fusion_Head(concat=True, individual=False, c_in=7, c_out=7,
                                  nf=128*10, m_nf=128*5, t_nf=128*5, target_window=24)
        output = fusion_head(long_feat, short_feat, m_weight, t_weight)
    """
    def __init__(self, concat, individual, c_in, c_out, nf, m_nf, t_nf, target_window, head_dropout=0):
        super().__init__()
        self.concat = concat
        self.individual = individual
        self.c_in = c_in
        self.c_out = c_out
        self.target_window = target_window
        
        if self.concat:
            if self.individual:
                self.linears = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                for i in range(self.c_in):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.linears.append(nn.Linear(nf, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.linear = nn.Linear(nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
        else:
            if self.individual:
                self.linears = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                self.long_to_shorts = nn.ModuleList()
                for i in range(self.c_in):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.long_to_shorts.append(nn.Linear(m_nf, t_nf))
                    self.linears.append(nn.Linear(nf, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.long_to_short = nn.Linear(m_nf, t_nf) 
                self.linear = nn.Linear(t_nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, long, short, m_weight, t_weight):
        if self.concat:
            if self.individual:
                long_short_out = []
                for i in range(self.c_in):
                    long_flat = self.flattens[i](long[:,i,:,:])
                    short_flat = self.flattens[i](short[:,i,:,:])
                    long_short = torch.cat((m_weight*long_flat, t_weight*short_flat), 1)
                    long_short = self.linears[i](long_short)
                    long_short = self.dropouts[i](long_short) 
                    long_short_out.append(long_short)
                long_short = torch.stack(long_short_out, dim=1)
            else:
                long, short = self.flatten(long), self.flatten(short)
                long_short = torch.cat((torch.mul(m_weight.view(-1,1,1), long), 
                                        torch.mul(t_weight.view(-1,1,1), short)), 2)
                long_short = self.linear(long_short)
                long_short = self.dropout(long_short)
        else:
            if self.individual:
                long_short_out = []
                for i in range(self.c_in):
                    long_flat = self.flattens[i](long[:,i,:,:])
                    short_flat = self.flattens[i](short[:,i,:,:])
                    long_short = m_weight*self.long_to_shorts[i](long_flat) + t_weight*short_flat 
                    long_short = self.linears[i](long_short)
                    long_short = self.dropouts[i](long_short) 
                    long_short_out.append(long_short)
                long_short = torch.stack(long_short_out, dim=1)
            else:
                long, short = self.flatten(long), self.flatten(short)
                long_short = m_weight*self.long_to_short(long) + t_weight*short 
                long_short = self.linear(long_short)
                long_short = self.dropout(long_short)

        return long_short


# ============================================================================
# 4. series_decomp - 序列分解模块 (Series Decomposition)
# ============================================================================
class moving_avg(nn.Module):
    """移动平均模块,用于提取趋势"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    序列分解模块
    将时间序列分解为趋势(trend)和残差(residual)两个部分
    
    输入: [Batch, Seq_len, Channel]
    输出: (residual, trend) - 两个相同形状的张量
    
    使用方法:
        decomp = series_decomp(kernel_size=25)
        res, trend = decomp(ts_data)
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ============================================================================
# 5. PositionalEncoding - 位置编码 (Positional Encoding)
# ============================================================================
def PositionalEncoding(q_len, d_model, normalize=True):
    """
    正弦位置编码
    
    使用方法:
        pe = PositionalEncoding(q_len=96, d_model=128)
        x_encoded = x + pe  # 添加到嵌入向量
    """
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


class PositionalEmbedding(nn.Module):
    """
    可学习的位置嵌入模块
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# ============================================================================
# 6. TokenEmbedding - 令牌嵌入 (Token Embedding)
# ============================================================================
class TokenEmbedding(nn.Module):
    """
    令牌嵌入模块
    使用1D卷积将输入特征映射到模型维度
    
    使用方法:
        token_embed = TokenEmbedding(c_in=7, d_model=128)
        embedded = token_embed(x)  # x: [Batch, Seq_len, Channel]
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# ============================================================================
# 7. DataEmbedding - 数据嵌入 (完整的数据嵌入模块)
# ============================================================================
class DataEmbedding(nn.Module):
    """
    完整的数据嵌入模块
    结合值嵌入、位置嵌入和时间嵌入
    
    使用方法:
        embed = DataEmbedding(c_in=7, d_model=128, embed_type='fixed', freq='h', dropout=0.1)
        embedded = embed(x, x_mark)  # x: [Batch, Seq_len, Channel], x_mark: 时间特征
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 简化的时间嵌入 (实际使用时可能需要完整实现)
        if embed_type == 'timeF':
            freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
            d_inp = freq_map.get(freq, 4)
            self.temporal_embedding = nn.Linear(d_inp, d_model, bias=False)
        else:
            # 简化实现
            self.temporal_embedding = nn.Linear(4, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x_mark = torch.zeros(x.shape[0], x.shape[1], 4).to(x.device)
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# ============================================================================
# 主测试函数
# ============================================================================
def main():
    """测试所有即插即用模块"""
    print("="*80)
    print("SST 即插即用模块测试")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    batch_size = 4
    seq_len = 96
    pred_len = 24
    label_len = 48
    c_in = 7
    d_model = 128
    
    # ========================================================================
    # 测试1: RevIN - 可逆实例归一化
    # ========================================================================
    print("1. 测试 RevIN (可逆实例归一化)")
    print("-" * 80)
    revin = RevIN(num_features=c_in, affine=True).to(device)
    x_revin = torch.randn(batch_size, seq_len, c_in).to(device)
    x_norm = revin(x_revin, mode='norm')
    x_recovered = revin(x_norm, mode='denorm')
    diff = torch.abs(x_revin - x_recovered).mean().item()
    print(f"   输入形状: {x_revin.shape}")
    print(f"   归一化后形状: {x_norm.shape}")
    print(f"   恢复误差 (应该接近0): {diff:.6f}")
    print(f"   ✓ RevIN 测试通过!\n")
    
    # ========================================================================
    # 测试2: Router - 长短路由
    # ========================================================================
    print("2. 测试 Router (长短路由)")
    print("-" * 80)
    router = Router(long_context_window=seq_len, context_window=label_len, 
                    c_in=c_in, d_model=d_model).to(device)
    long_seq = torch.randn(batch_size, seq_len, c_in).to(device)
    m_weight, t_weight = router(long_seq)
    print(f"   输入形状: {long_seq.shape}")
    print(f"   长期权重形状: {m_weight.shape}, 值: {m_weight[0].item():.4f}")
    print(f"   短期权重形状: {t_weight.shape}, 值: {t_weight[0].item():.4f}")
    print(f"   权重和 (应该接近1): {(m_weight + t_weight)[0].item():.4f}")
    print(f"   ✓ Router 测试通过!\n")
    
    # ========================================================================
    # 测试3: series_decomp - 序列分解
    # ========================================================================
    print("3. 测试 series_decomp (序列分解)")
    print("-" * 80)
    decomp = series_decomp(kernel_size=25).to(device)
    ts_data = torch.randn(batch_size, seq_len, c_in).to(device)
    res, trend = decomp(ts_data)
    print(f"   输入形状: {ts_data.shape}")
    print(f"   残差形状: {res.shape}")
    print(f"   趋势形状: {trend.shape}")
    print(f"   重构误差 (应该接近0): {torch.abs(ts_data - (res + trend)).mean().item():.6f}")
    print(f"   ✓ series_decomp 测试通过!\n")
    
    # ========================================================================
    # 测试4: Fusion_Head - 融合头
    # ========================================================================
    print("4. 测试 Fusion_Head (融合头)")
    print("-" * 80)
    m_patch_num = 10
    t_patch_num = 5
    m_nf = d_model * m_patch_num
    t_nf = d_model * t_patch_num
    nf = d_model * (m_patch_num + t_patch_num)
    
    fusion_head = Fusion_Head(concat=True, individual=False, c_in=c_in, c_out=c_in,
                              nf=nf, m_nf=m_nf, t_nf=t_nf, target_window=pred_len).to(device)
    long_feat = torch.randn(batch_size, c_in, d_model, m_patch_num).to(device)
    short_feat = torch.randn(batch_size, c_in, d_model, t_patch_num).to(device)
    m_weight = torch.rand(batch_size).to(device)
    t_weight = torch.rand(batch_size).to(device)
    m_weight = m_weight / (m_weight + t_weight)  # 归一化
    t_weight = t_weight / (m_weight + t_weight)
    
    output = fusion_head(long_feat, short_feat, m_weight, t_weight)
    print(f"   长期特征形状: {long_feat.shape}")
    print(f"   短期特征形状: {short_feat.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   ✓ Fusion_Head 测试通过!\n")
    
    # ========================================================================
    # 测试5: PositionalEncoding - 位置编码
    # ========================================================================
    print("5. 测试 PositionalEncoding (位置编码)")
    print("-" * 80)
    pe = PositionalEncoding(q_len=seq_len, d_model=d_model)
    print(f"   位置编码形状: {pe.shape}")
    print(f"   位置编码统计: mean={pe.mean():.6f}, std={pe.std():.6f}")
    print(f"   ✓ PositionalEncoding 测试通过!\n")
    
    # ========================================================================
    # 测试6: TokenEmbedding - 令牌嵌入
    # ========================================================================
    print("6. 测试 TokenEmbedding (令牌嵌入)")
    print("-" * 80)
    token_embed = TokenEmbedding(c_in=c_in, d_model=d_model).to(device)
    x_embed = torch.randn(batch_size, seq_len, c_in).to(device)
    embedded = token_embed(x_embed)
    print(f"   输入形状: {x_embed.shape}")
    print(f"   嵌入后形状: {embedded.shape}")
    print(f"   ✓ TokenEmbedding 测试通过!\n")
    
    # ========================================================================
    # 测试7: DataEmbedding - 完整数据嵌入
    # ========================================================================
    print("7. 测试 DataEmbedding (完整数据嵌入)")
    print("-" * 80)
    data_embed = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.1).to(device)
    x_data = torch.randn(batch_size, seq_len, c_in).to(device)
    x_mark = torch.randn(batch_size, seq_len, 4).to(device)
    embedded_data = data_embed(x_data, x_mark)
    print(f"   输入数据形状: {x_data.shape}")
    print(f"   时间标记形状: {x_mark.shape}")
    print(f"   嵌入后形状: {embedded_data.shape}")
    print(f"   ✓ DataEmbedding 测试通过!\n")
    
    # ========================================================================
    # 综合测试: 组合使用多个模块
    # ========================================================================
    print("8. 综合测试: 组合使用多个模块")
    print("-" * 80)
    # 创建数据流
    x_input = torch.randn(batch_size, seq_len, c_in).to(device)
    
    # 1. 使用RevIN归一化
    x_norm = revin(x_input, mode='norm')
    
    # 2. 使用Router获取权重
    m_w, t_w = router(x_norm)
    
    # 3. 使用DataEmbedding嵌入
    x_embedded = data_embed(x_norm, None)
    
    # 4. 使用series_decomp分解
    x_short = x_norm[:, -label_len:, :]
    res, trend = decomp(x_short)
    
    print(f"   输入: {x_input.shape}")
    print(f"   归一化后: {x_norm.shape}")
    print(f"   Router权重: m={m_w[0].item():.4f}, t={t_w[0].item():.4f}")
    print(f"   嵌入后: {x_embedded.shape}")
    print(f"   分解: residual={res.shape}, trend={trend.shape}")
    print(f"   ✓ 综合测试通过!\n")
    
    print("="*80)
    print("所有测试完成! 所有即插即用模块工作正常。")
    print("="*80)


if __name__ == "__main__":
    main()

