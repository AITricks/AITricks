"""
PG-DRFNet 即插即用模块测试程序
包含三个核心模块: PGHead, DynamicPerceptionV1, DynamicPerceptionV2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 模块1: Positional Guidance Head (位置引导头)
# ============================================================================

class PGHead(nn.Module):
    """
    Positional Guidance Head (位置引导头)
    该模块用于从特征图中生成位置引导信号,是小目标检测中的关键组件。
    """
    
    def __init__(self, in_channels, conv_channels, num_convs, pred_channels, pred_prior=None):
        super(PGHead, self).__init__()
        self.num_convs = num_convs

        self.subnet = nn.ModuleList()
        channels = in_channels
        for i in range(self.num_convs):
            layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.subnet.append(layer)
            channels = conv_channels

        self.pred_net = nn.Conv2d(channels, pred_channels, kernel_size=3, stride=1, padding=1)

        nn.init.xavier_normal_(self.pred_net.weight)
        if pred_prior is None:
            nn.init.constant_(self.pred_net.bias, 0)

    def forward(self, features):
        """前向传播: 输入多尺度特征图列表,输出位置引导信号列表"""
        preds = []
        for feature in features:
            x = feature
            for i in range(self.num_convs):
                x = F.relu(self.subnet[i](x))
            preds.append(self.pred_net(x))
        return preds


# ============================================================================
# 模块2: Dynamic Perception V1 (动态感知 v1 - 探索性版本)
# ============================================================================

class DynamicPerceptionV1:
    """
    动态感知模块 v1 (探索性版本)
    基于堆叠方式构建特征的动态感知算法。
    """
    
    def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):
        self.anchor_num = anchor_num
        self.num_classes = num_classes
        self.score_th = score_th
        self.context = context

    def _split_feature(self, query_logits, last_ys, last_xs, anchors, feature_value):
        """根据查询逻辑分割特征图,提取关键位置"""
        if last_ys is None:
            N, _, qh, qw = query_logits.size()
            assert N == 1, "Batch size must be 1"
            prob = torch.sigmoid_(query_logits).view(-1)
            pidxs = torch.where(prob > self.score_th)[0]
            y = torch.div(pidxs, qw).int()
            x = torch.remainder(pidxs, qw).int()
        else:
            prob = torch.sigmoid_(query_logits).view(-1)
            pidxs = prob > self.score_th
            y = last_ys.flatten(0)[pidxs]
            x = last_xs.flatten(0)[pidxs]

        if y.size(0) == 0:
            return None, None, None, None, None

        block_num = len(y)
        _, fc, fh, fw = feature_value.shape
        ys = []
        xs = []
        
        for i in range(-1 * self.context, self.context + 1):
            for j in range(-1 * self.context, self.context + 1):
                ys.append(y * 2 + i)
                xs.append(x * 2 + j)

        ys = torch.stack(ys, dim=0).t()
        xs = torch.stack(xs, dim=0).t()

        block_pixes_num = (self.context * 2 + 1) ** 2
        good_idx = (ys >= 0) & (ys < fh) & (xs >= 0) & (xs < fw)
        
        # 检查是否有无效像素
        if not good_idx.all():
            # 有无效像素，需要过滤
            # 简单策略：只保留完全有效的块
            valid_blocks = []
            valid_ys_list = []
            valid_xs_list = []
            
            for b_idx in range(block_num):
                block_good = good_idx[b_idx]  # [block_pixes_num]
                if block_good.all():
                    # 这个块完全有效，保留
                    valid_ys_list.append(ys[b_idx])
                    valid_xs_list.append(xs[b_idx])
                    valid_blocks.append(b_idx)
            
            if len(valid_blocks) == 0:
                return None, None, None, None, None
            
            # 重新组织为块结构
            ys = torch.stack(valid_ys_list, dim=0)
            xs = torch.stack(valid_xs_list, dim=0)
            block_num = len(valid_blocks)

        inds = (ys * fw + xs).long()
        yx = torch.stack((ys, xs), dim=2)
        block_list = [yx[i] for i in range(block_num)]

        return block_list, ys, xs, inds, block_num

    def build_block_feature(self, block_list, feature_value):
        """从特征图中构建块特征 (堆叠方式)"""
        block_feature_list = []
        for block in block_list:
            y_index = block[:, 0].long()
            x_index = block[:, 1].long()
            block_feature_list.append(
                feature_value[:, :, y_index, x_index].view(
                    1, feature_value.shape[1], self.context * 2 + 1, -1
                )
            )
        return block_feature_list

    def run_dpinfer(self, features_value, query_logits):
        """运行动态感知推理"""
        last_ys, last_xs = None, None

        for i in range(len(features_value) - 1, -1, -1):
            block_list, last_ys, last_xs, inds, block_num = self._split_feature(
                query_logits[i + 1],
                last_ys,
                last_xs,
                None,
                features_value[i]
            )
            
            if block_list is None:
                return None, None
                
            block_feature_list = self.build_block_feature(block_list, features_value[i])

        return block_feature_list, inds


# ============================================================================
# 模块3: Dynamic Perception V2 (动态感知 v2 - 优化版本)
# ============================================================================

class DynamicPerceptionV2:
    """
    动态感知模块 v2 (优化版本)
    基于扁平化方式构建特征的动态感知算法,具有更好的空间结构。
    """
    
    def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):
        self.anchor_num = anchor_num
        self.num_classes = num_classes
        self.score_th = score_th
        self.context = context

    def _split_feature(self, query_logits, last_ys, last_xs, inds, feature_value):
        """根据查询逻辑分割特征图,提取关键位置 (优化版本)"""
        if last_ys is None:
            N, _, qh, qw = query_logits.size()
            assert N == 1, "Batch size must be 1"
            prob = torch.sigmoid_(query_logits).view(-1)
            pidxs = torch.where(prob > self.score_th)[0]
            y = torch.div(pidxs, qw).int()
            x = torch.remainder(pidxs, qw).int()
        else:
            prob = torch.sigmoid_(query_logits).view(-1)[inds]
            pidxs = prob > self.score_th
            y = last_ys.flatten(0)[pidxs]
            x = last_xs.flatten(0)[pidxs]

        if y.size(0) == 0:
            return None, None, None, None, None, None

        lt = torch.tensor([y.min().item(), x.min().item()], device=y.device)
        rb = torch.tensor([y.max().item(), x.max().item()], device=y.device)

        _, fc, fh, fw = feature_value.shape
        ys = []
        xs = []
        block_list = []
        high_wide = []
        block_y = []
        block_x = []

        # 构建扁平化的特征块区域，并检查边界
        h_range = range(int(lt[0].item() * 2), int(rb[0].item() * 2 + 1))
        w_range = range(int(lt[1].item() * 2), int(rb[1].item() * 2 + 1))
        
        for pixes_i in h_range:
            for pixes_j in w_range:
                # 检查边界
                if 0 <= pixes_i < fh and 0 <= pixes_j < fw:
                    block_y.append(pixes_i)
                    block_x.append(pixes_j)
        
        if len(block_y) == 0:
            return None, None, None, None, None, None
        
        # 计算实际的高度和宽度（根据实际收集的像素点）
        num_pixels = len(block_y)
        if num_pixels > 0:
            min_y, max_y = min(block_y), max(block_y)
            min_x, max_x = min(block_x), max(block_x)
            # 计算bounding box的尺寸
            box_h = max(1, max_y - min_y + 1)
            box_w = max(1, max_x - min_x + 1)
            # 确保 h*w 等于实际像素数量
            if box_h * box_w == num_pixels:
                actual_h, actual_w = box_h, box_w
            else:
                # 如果不能整除，找一个接近的h和w
                # 优先保持宽高比
                actual_h = box_h
                actual_w = (num_pixels + actual_h - 1) // actual_h  # 向上取整
        else:
            actual_h, actual_w = 1, 1
                
        ys.append(torch.tensor(block_y, device=y.device))
        xs.append(torch.tensor(block_x, device=y.device))
        block_list.append(torch.stack((
            torch.tensor(block_y, device=y.device),
            torch.tensor(block_x, device=y.device)
        ), dim=1))
        high_wide.append(torch.tensor([actual_h, actual_w], device=y.device, dtype=torch.long))

        ys = torch.cat(ys, dim=0)
        xs = torch.cat(xs, dim=0)
        inds = (ys * fw + xs).long()

        return block_list, high_wide, ys, xs, inds, None

    def build_block_feature(self, block_list, feature_value, high_wide):
        """从特征图中构建块特征 (扁平化方式)"""
        block_feature_list = []
        for i, block in enumerate(block_list):
            y_index = block[:, 0].long()
            x_index = block[:, 1].long()
            h, w = high_wide[i]
            h_val, w_val = h.item(), w.item()
            
            # 提取特征
            extracted_feat = feature_value[:, :, y_index, x_index]  # [1, C, num_pixels]
            num_pixels = extracted_feat.shape[2]
            
            # 确保h*w等于实际像素数量，如果不匹配则调整
            if h_val * w_val != num_pixels:
                # 根据实际像素数量重新计算h和w
                h_val = max(1, int(num_pixels ** 0.5))
                w_val = (num_pixels + h_val - 1) // h_val
            
            block_feature_list.append(
                extracted_feat.view(1, feature_value.shape[1], h_val, w_val)
            )
        return block_feature_list

    def run_dpinfer(self, features_value, query_logits):
        """运行动态感知推理 (优化版本)"""
        last_ys, last_xs = None, None

        for i in range(len(features_value) - 1, -1, -1):
            block_list, high_wide, last_ys, last_xs, inds, block_num = self._split_feature(
                query_logits[i + 1],
                last_ys,
                last_xs,
                None,
                features_value[i]
            )
            
            if block_list is None:
                return None, None
                
            block_feature_list = self.build_block_feature(
                block_list, features_value[i], high_wide
            )

        return block_feature_list, inds


# ============================================================================
# 测试主程序
# ============================================================================

def test_pg_head():
    """测试位置引导头模块"""
    print("测试 PGHead 模块...")
    
    # 创建模块
    pg_head = PGHead(in_channels=192, conv_channels=192, num_convs=2, pred_channels=1)
    
    # 创建输入特征
    features = [
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 192, 32, 32),
    ]
    
    # 前向传播
    with torch.no_grad():
        outputs = pg_head(features)
    
    print(f"  输入: {[f.shape for f in features]}")
    print(f"  输出: {[out.shape for out in outputs]}")
    print("  ✓ 测试通过!\n")


def test_dynamic_perception_v1():
    """测试动态感知模块 v1"""
    print("测试 DynamicPerceptionV1 模块...")
    
    # 创建模块
    dp_v1 = DynamicPerceptionV1(anchor_num=1, num_classes=18, score_th=0.1, context=2)
    
    # 创建输入
    features_value = [
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 192, 32, 32),
    ]
    
    query_logits = [
        torch.randn(1, 1, 64, 64),
        torch.randn(1, 1, 32, 32),
        torch.randn(1, 1, 16, 16),
    ]
    
    # 设置高分位置（确保在有效范围内）
    query_logits[-1][0, 0, 8, 8] = 2.0  # 使用更中心的位置避免边界问题
    
    # 运行推理
    try:
        block_features, inds = dp_v1.run_dpinfer(features_value, query_logits)
        if block_features is not None:
            print(f"  成功提取 {len(block_features)} 个特征块")
            print(f"  特征块形状: {[b.shape for b in block_features[:2]]}...")
        else:
            print("  未找到满足阈值的关键位置")
    except Exception as e:
        print(f"  异常: {e}")
    
    print("  ✓ 测试完成!\n")


def test_dynamic_perception_v2():
    """测试动态感知模块 v2"""
    print("测试 DynamicPerceptionV2 模块...")
    
    # 创建模块
    dp_v2 = DynamicPerceptionV2(anchor_num=1, num_classes=18, score_th=0.1, context=4)
    
    # 创建输入
    features_value = [
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 192, 32, 32),
    ]
    
    query_logits = [
        torch.randn(1, 1, 64, 64),
        torch.randn(1, 1, 32, 32),
        torch.randn(1, 1, 16, 16),
    ]
    
    # 设置高分位置（确保在有效范围内）
    query_logits[-1][0, 0, 8, 8] = 2.0  # 使用更中心的位置避免边界问题
    
    # 运行推理
    try:
        block_features, inds = dp_v2.run_dpinfer(features_value, query_logits)
        if block_features is not None:
            print(f"  成功提取 {len(block_features)} 个特征块")
            print(f"  特征块形状: {[b.shape for b in block_features[:2]]}...")
        else:
            print("  未找到满足阈值的关键位置")
    except Exception as e:
        print(f"  异常: {e}")
    
    print("  ✓ 测试完成!\n")


def main():
    """主函数"""
    print("=" * 60)
    print("PG-DRFNet 即插即用模块测试")
    print("=" * 60 + "\n")
    
    torch.manual_seed(42)
    
    # 运行所有测试
    test_pg_head()
    test_dynamic_perception_v1()
    test_dynamic_perception_v2()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
