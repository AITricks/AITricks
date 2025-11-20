# plug_and_play_modules.py 说明

`plug_and_play_modules.py` 汇集了 ConDSeg 框架中可复用的「即插即用」模块，涵盖注意力机制、特征增强、特征解耦、辅助预测以及 CDFA 核心聚合模块等，可与不同 backbone（如 `network/model.py` 中的 ResNet、`network_pvt/model.py` 中的 PVTv2）配合使用。

## 模块概览

- **CBR**：`Conv2d + BatchNorm2d + ReLU` 基础块，广泛作为子模块。
- **channel_attention / spatial_attention**：通道与空间注意力，组合实现 CBAM 风格的重加权。
- **dilated_conv**：特征增强模块（FEM），包含多尺度空洞卷积串联通道注意力，再经融合与空间注意力输出。
- **DecoupleLayer**：将高层特征解耦为前景、背景、不确定性三个分支，供后续模块使用。
- **AuxiliaryHead**：三个上采样预测头，生成前景/背景/不确定性掩膜，可在训练中提供辅助监督。
- **ContrastDrivenFeatureAggregation (CDFA)**：核心聚合器，利用前景/背景特征指导主特征的对比注意力加权。
- **CDFAPreprocess**：用于将解耦特征调整到不同尺度（逐次上采样 + CBR）。
- **测试函数**：`test_*` 与 `test_integration` 便于单独或整体验证模块正确性。

## 依赖

- PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
- Python 内置 `math`

确保 GPU 可用时自动切换至 CUDA，若无则回退到 CPU。

## 快速使用示例

1. **特征解耦**
   ```python
   decouple = DecoupleLayer(in_c=1024, out_c=128)
   f_fg, f_bg, f_uc = decouple(x4)  # x4 为主干高层特征
   ```
2. **多尺度预处理 + 特征增强**
   ```python
   preprocess_fg3 = CDFAPreprocess(128, 128, up_scale=2)
   f_fg3 = preprocess_fg3(f_fg)

   dconv3 = dilated_conv(512, 128)
   d3 = dconv3(x3)  # x3 为 backbone 中级特征
   ```
3. **CDFA 聚合**
   ```python
   cdfa3 = ContrastDrivenFeatureAggregation(
       in_c=128, dim=128, num_heads=4, kernel_size=3, padding=1, stride=1
   )
   fused = cdfa3(d3, f_fg3, f_bg3)
   ```
4. **辅助头监督**
   ```python
   aux_head = AuxiliaryHead(in_c=128)
   mask_fg, mask_bg, mask_uc = aux_head(f_fg, f_bg, f_uc)
   ```

## 集成测试

执行 `python plug_and_play_modules.py` 将依次运行：

1. 单元测试：`test_cdfa_module`, `test_decouple_layer`, `test_dilated_conv`, `test_auxiliary_head`, `test_attention_modules`
2. 集成测试：`test_integration`，模拟从 backbone 输出到 CDFA 及辅助头的完整流程

全部通过后会打印 “✓ 集成测试通过! 所有即插即用模块工作正常”。

## 自定义 / 扩展建议

- **更换 backbone**：保持特征维度与尺度对应即可复用这些模块。
- **调整注意力头数**：`ContrastDrivenFeatureAggregation` 的 `num_heads` 与 `dim` 需整除关系。
- **不同上采样需求**：`CDFAPreprocess` 通过 `up_scale` 自动计算上采样次数，可轻松适配更多层级。

该文件中的模块均为独立类，可按需导入、组合，为其他分割或检测项目提供高效的前景/背景对比增强能力。

