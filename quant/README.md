# eDQA 复现骨架

按论文《eDQA: Efficient Deep Quantization of DNN Activations on Edge Devices》的方法论拆分的模块化实现。每个模块对应一个文件。

## 模块 → 论文对应

| 文件 | 中文 | 对应论文 |
|---|---|---|
| `quantizer.py` | 量化基元 (core) | Eq.(1)、对称量化、右移 + 移位误差表 |
| `edqa_layer.py` | 层量化 / 反量化 | Algorithm 1、Algorithm 2 |
| `ranking.py` | 离线贪心排名 | Algorithm 3 (IQ 搜索) |
| `compression.py` | 压缩模块 | Table 1 的 Huffman / Deflate / LZMA / ZSTD |
| `hooks.py` | 激活钩子 | 只量化激活的 fake-quant 基础设施 |
| `data.py` | 数据集 | CIFAR-10 / TinyImageNet + 校准子集 |
| `baselines.py` | 基线 | Direct / PoT / NoisyQuant |
| `evaluate.py` | 评估框架 | Table 2、Figure 3/4、Table 1 |
| `run_experiments.py` | 端到端示例 | 把上面串起来的模板 |

## 数据流

离线：`ranking.rank_channels` 用小校准子集跑贪心搜索 → 得到每层通道排名表 `R`（缓存到磁盘）。
推理：`hooks.QuantManager` 在每层激活输出上套 `edqa_fake_quantize`（Algorithm 1+2），用 `R` 和比例 `r` 决定哪些通道走"多 m 位 + 右移 + 压缩误差"的路径。
评估：`evaluate.*` 复现论文的表和图。

## 复现时最容易踩的坑

1. **scale 是逐通道 (per-channel) 的 `|max|`**，不是整层共用一个标量。这点跟论文作者直接确认过（2026-07 邮件："And yes, it's per channel quantization."）——Algorithm 1/2 的 `|max(A_layer)|` 字面写法容易读成整层共享，实际不是。见 `quantizer.compute_scale` 的 `channel_dim` 参数（默认 `1`=逐通道；传 `None` 可切回旧的整层字面实现，用于对比）。
2. **论文分母用 `2^(N-1)`**（非标准的 `2^(N-1)-1`）。想切换在 `quantizer.SCALE_DENOM_MINUS_ONE` 改。
3. **移位误差语义**：读取低 m 位整数 `k` → 小数 `k/2^m`（落在 [0,1)）。反量化是 `Δ_N·(code + se)`，不是 `Δ_{N+m}·q_more`；两者等价，见 `quantizer.right_shift_with_error` 的推导注释。m=3 时误差取值 {0,0.125,…,0.875}，正好是 Figure 2 横轴。
4. **反量化必须携带 scale**：反量化时原始激活已丢失，`max_abs` 要从量化阶段传回，不能重新算。
5. **Algorithm 3 每评估一个通道 = 一次完整推理**，O(L·C) 次。务必用小校准子集（CIFAR-10 5000、TinyImageNet 2500），必要时用 `channel_subsample` 只评估部分候选通道。`ranking.rank_channels` 对同一个 calibration loader 反复 fresh-iterate 数百次，`data.calibration_loader` 默认 `num_workers=0`——改成 >0 在 Windows 上会因为反复 spawn worker 进程而慢 4 倍以上。
6. **"重要通道"= 跳过量化后精度更高的通道**。排名阶段用的是 Direct + skip 模式（`hooks` 里的 `direct` 模式），不是完整 eDQA。
7. **ViT 的 "channel" 需要你自己定义**。默认 `channel_dim=1`（卷积 NCHW / Linear 的特征维）。对 Transformer 用 `hooks.vit_mlp_target_layers`（只 hook 每个 block 的 MLPBlock 输出，而不是每个 Linear），配合 `channel_dim=2`（形状是 `(B, N_tokens, hidden_dim)`，channel 在第 2 维不是第 1 维）。
8. **NoisyQuant 是简化版**（`baselines.fake_quantize_noisyquant`）——用轻量随机搜索替代作者的离线结构化搜索。要对齐论文数值需换成作者放出的搜索实现。
9. **压缩算法的选择不影响精度评估**：所有压缩器都是无损的，`errors[channel]` 解压后与压缩前逐位相同，所以 `compare_methods`/`sweep_ratio`/`sweep_extra_bits`（只关心精度）默认用 `compression.IdentityCompressor`（无操作，仅 `.tobytes()`），不是 `huffman`。这不是为了图快而牺牲精度——精度数字和用哪个压缩器完全无关，只有 `compression_table`（Table 1，专门测压缩比/延迟）才需要真实编解码器。之前误把 Huffman 当默认值用在精度评估上时，纯 Python 的逐元素编解码占了单次评估 90% 以上的时间（profile 过，8 batch 从 440s 降到 17.6s）。
10. **per-layer rank 的接线**：`evaluate` 里演示了从 `ranks` 字典按层名取 rank。`compression_table` 里的 `_any_rank` 是个回退占位，真跑时要把每层正确的 rank 线程进去。
11. **（已解决）曾经的复现缺口：3-bit 直接量化在深层网络里比论文报的崩得更狠。** 早期用字面的 `Δ_N=|max(A_layer)|/2^(N-1)`（整层共享一个 scale）实现时，ResNet-18/TinyImageNet 上 3-bit Direct/eDQA 精度崩到接近随机猜测（0.49%），远低于论文的 49.06%/63.61%。排查过程：先确认不是量化数学的 bug（5-bit/8-bit 精度能平滑恢复到接近全精度），也不是训练不足（全精度 eval 精度和训练 val_acc 吻合）；再确认根因是 scale 被极少数离群激活值主导——`layer1.0.relu` 单独量化就能把精度从 76% 砸到 8%，该层真实 max 是 99 百分位数的 4.25 倍，导致 84% 的正常值被量化round 到 0。**后来跟论文作者确认，scale 实际是逐通道 (per-channel) 算的，不是整层共用**（见坑 #1）——每个通道用自己的 max，不会被别的通道的离群值拖累。改成逐通道后，实测同类合成数据上 Direct 3-bit 的平均误差从 0.25 降到 0.04（降了 5.8 倍），量级上跟诊断阶段估算的"离群值问题需要 ~4-5 倍分辨率提升才能解决"吻合。

12. **（2026-07-18 发现）纯"能量"（per-channel 激活平方和）排序，比论文自己的 Algorithm 3 贪心搜索、也比 SVD 矩阵秩 surrogate 都更准更稳——但只在 ResNet-18/TinyImageNet 上如此，见坑 #13，不能当普适结论。** ResNet-18/TinyImageNet, eDQA 3-bit, clip_p999, 3 个 calib_seed 全测试集验证：纯 rank-direction surrogate mean=63.31%/std=1.77pp，full-scan 贪心 mean=64.29%/std=0.63pp，纯 energy（`rank_surrogates.rank_channels_via_energy`）mean=66.51%/std=0.20pp——比论文本身的数字（63.61%）高出近 3pp，且几乎不需要计算（一次校准前向即可，没有 SVD、没有逐层方向搜索、更没有 O(L·C) 全量评估）。中途还试过一个"rank 方向搜索的 margin 太小就 fallback 到 energy"的 cascade 方案，结果和纯 energy 几乎完全一样（3 个 seed 里除 1 层外全部 fallback 到 energy）——说明 margin 门控本身没有额外价值，直接用纯 energy 就够了。已接入 `run_experiments_tuned.py --ranking energy`。

13. **（2026-07-19 发现）"能量"排序不是普适更优，效果依赖模型/架构——目前没有哪种便宜 surrogate（energy 或矩阵秩）能稳定替代论文的贪心搜索。** 跨模型对比（都是 clip_p999, eDQA 3-bit, 全测试集）：

    | 模型/数据集 | greedy | energy | 矩阵秩 surrogate |
    |---|---|---|---|
    | ResNet-18/TinyImageNet（3 seed） | mean=64.29%, std=0.63pp | **mean=66.51%, std=0.20pp（赢）** | mean=63.31%, std=1.77pp |
    | ResNet-32/CIFAR-10（3 seed） | **mean=89.19%, std=0.36pp（赢）** | mean=87.45%, std=0.51pp | mean=86.90%, std=0.33pp |
    | MobileNetV2/CIFAR-10（单 seed） | 86.22% | **91.36%（赢，+5.14pp）** | 未跑 |

    ResNet-32 和 ResNet-18 的差异里，图像分辨率（32×32 vs 224×224）和通道宽度（最宽64 vs 最宽512）本来是**同步变化、互相混淆的**——两个数据点分不清到底哪个是驱动因素。MobileNetV2/CIFAR-10（低分辨率 32×32 + 超宽通道最宽1280）是解耦这两个变量的天然对照实验：结果是 **energy 大赢**（+5.14pp，比 ResNet-18 那次的优势还大）——表现像通道宽的 ResNet-18，不像同样低分辨率的 ResNet-32。**这次解耦实验的结果指向"通道宽度"才是关键因素，"分辨率驱动冗余度"这个假说没有被支持**（样本量仍只有 3 个模型，非定论，只是这一次对照没有支持该假说）。矩阵秩 surrogate 在 MobileNetV2 上还没跑，三方对比表暂时缺这一格。

    速度上 energy 稳赢、跟模型无关（O(1) vs greedy 的 O(L·C)）——MobileNetV2 上 greedy full-scan 排序本身预计要 80+ 分钟，energy 排序只要 0.5 秒；ResNet-18 上 greedy full-scan 排序要 100+ 分钟/seed，energy 同样只要不到 1 秒。但速度优势不等于精度优势，两者要分开评估，见上表。

    **"能量"（per-channel 激活平方和）作为重要性指标的机制解释**：一个通道对下游层输出的贡献量级，大致正比于它自身激活幅值 × 下游权重；长期输出接近零的通道对下游计算贡献小，量化它伤害也小。这跟剪枝文献里"权重 L1 范数大的 filter 更重要"（Li et al., *Pruning Filters for Efficient ConvNets*, ICLR 2017）是同一直觉，只是我们测的是激活（真实数据跑出来的响应）而不是权重本身。相关文献：feature-map L1/L2/L∞-norm 直接作为通道重要性判据（arXiv:1812.03608）；APoZ，用 ReLU 后激活为零的比例判断神经元冗余（Hu et al., *Network Trimming*, arXiv:1607.03250）；用 feature map 核范数（SVD 奇异值之和）做"能量"度量剪枝（Yeom et al., *Toward Compact Deep Neural Networks via Energy-Aware Pruning*, arXiv:2103.10858，同时呼应我们 energy 和矩阵秩两条线）。

    energy 跟 greedy 的差距，大概率来自 energy 测不到的两块东西：(a) **冗余度**——两个通道响应模式高度相关时，只保护一个精度就基本不掉，energy 纯看幅值会把两个都判定重要，greedy 的 skip-test 能测出这种冗余；(b) **下游"杠杆"**——通道幅值不大但连接到下游关键决策路径时，实际影响可能远超其幅值暗示的量级。**还没有直接测过这两点**（比如没算过任何层的 pairwise 通道相关系数），是有依据的推测，不是已验证结论。低成本验证方法：把某层激活 reshape 成 `[C, N*H*W]`，一次矩阵乘法算出 C×C 相关系数矩阵，看平均 pairwise 相关度（或者该矩阵的有效秩）是否跟"energy 相对 greedy 的输赢"对得上——**这个还没做，是下一步可以补的实验**。

14. **（2026-07-20 修复 + 2026-07-21 结论）ViT 的 hook 覆盖率 bug，以及修复后 Fig3/Fig4 扫描出的两条规律。** `vit_mlp_target_layers`（旧）只 hook 每个 transformer block 最终的 MLP 输出，12/37 个激活层，漏掉了全部 self-attention 输出和 3072 维的 MLP 中间激活（post-Linear、pre-GELU，正是 ViT 量化文献公认最难量化的地方）。换成 `vit_full_target_layers`（38 层：每个 block 的 self_attention + mlp.0 + mlp，加 conv_proj + heads.head，逐层 channel_dim）后，`QuantManager`/`MethodManager` 也相应加了 tuple-output 处理（`nn.MultiheadAttention` 返回 `(output, weights)`，直接 hook 在它上面时不能只认 Tensor）。修复后 Direct 3-bit 从 82.26%→76.81%（更容易掉分，方向正确），但离论文的 15.07% 暴跌仍差 61.7pp，根因未完全查明（详见 `quant/RESULTS.md` 已知问题）。

    修复后拿 energy 排序在全测试集上把 Fig3（r 扫描）、Fig4（m 扫描）跑了一遍，之前光跑没细看结论——实际上是有规律的：

    - **Fig3**：r=0.1(82.83%)→r=0.5(84.66%) 平均每 0.1 涨约 0.46pp；r=0.5→r=0.9（84.66%→85.03%，中间 r=0.7 峰值 85.22%）平均每 0.1 只涨约 0.09pp，边际收益缩到约 1/5。跟论文 Section 4.3 原话一致："impact of important ratios on accuracy is reduced...we find that it is 50% for both models"——但论文这句话说的是 ResNet-32/MobileNetV2，**论文自己从没在 ViT 或 ResNet-18 上跑过 Fig3/Fig4**，这次算是把这条规律扩展验证到了 transformer 架构上。
    - **Fig4**：m=1→m=2 涨 1.24pp（83.15%→84.39%），m=2→m=3 只涨 0.03pp（84.39%→84.42%），边际收益缩小约 40 倍，比论文原话"m=2 到 m=3 的提升明显小于 m=1 到 m=2"更夸张。同样支持论文自己的建议——m 没必要往大了加。
    - side note：r=0.7/0.8 时 eDQA(energy) 3-bit（85.22%/85.20%）**超过了全精度**（84.63%），量化偶尔轻微超过全精度不算罕见（有点像正则化效应），但这只是单次运行，别当结论用。

    另外验证了一个方法学顾虑：`rank_channels_via_energy` 默认 `calib_batches=1`，实际只用 128 张校准图（TinyImageNet 200 类，期望覆盖约 47%），greedy 用 512 张（期望覆盖约 92%）——担心这个覆盖率差异是 energy 赢 greedy 的 confound。把 `calib_batches` 提到 4（512 张，跟 greedy 同等覆盖）在 ResNet-18（3 seed）和 ViT（1 seed）上重新测：两边结果几乎不变（ResNet-18 66.51%→66.43% 均值；ViT 84.42%→84.45%），确认不是 confound。侧面证据：greedy 跨 calib_seed 的标准差（0.63pp）比 energy（0.20~0.40pp）明显大——energy 测的是激活量级，不看 label，天然对"具体校准到了哪些图片/类别"不敏感；greedy 靠准确率打分，直接依赖 label，天然更依赖类别覆盖。这个方差差异本身是"energy 类别无关"这个解释的独立证据。

    最后，ViT 的 greedy full-scan（Algorithm 3 字面全量搜索）目前判定不可行：用 Kaggle 上一次跑到一半的任务数据反推（layer0/768 通道耗时 100 分钟≈7.8 秒/候选），38 层全覆盖方案预估总耗时 **~122 小时**，超出 Kaggle 免费档单 session 12 小时上限一个数量级。论文自己用的是单卡 RTX 3090（Section 4.1，无多卡），按算力比例折算大概 ~30 小时——量级上更像是"作者就是跑了一两天"，不像用了什么特殊加速手段，但也没法排除作者对 ViT 的"layer"粒度定义比我们更粗（论文全文没有 ViT 专属的层定义细节）。这条线索前已决定先问作者，暂不用 subsample 凑数——subsample 到能避免过度退化的最小值（按各层 r=0.4 的实际 k）也要 ~49 小时，subsample 到能接受的时长（如 64，~5.3 小时）又会让最宽的 `mlp.0` 层（3072 通道，r=0.4 需要 k=1229）严重退化，两难，没有两全的折中点。

## 依赖

- 必需：`torch`, `torchvision`, `numpy`
- 可选：`zstandard`（ZSTD 压缩；缺了会给明确报错，其他三种压缩不受影响）

## 快速开始

```bash
python -m quant.run_experiments --experiment resnet32_cifar10        # 现成的第三方 CIFAR-10 checkpoint
python -m quant.run_experiments --experiment resnet18_tinyimagenet   # 需要先跑 model/finetune_tinyimagenet.py
python -m quant.run_experiments --experiment vit_b16_tinyimagenet    # 同上
python -m quant.run_experiments --experiment mobilenetv2_cifar10     # 现成的第三方 CIFAR-10 checkpoint
```

`run_experiments.EXPERIMENTS` 里每一项对应论文 Table 2 的一个 (模型, 数据集) 组合，`build_*` 函数负责加载对应的预训练/微调权重（eDQA 只量化激活，权重必须已经训练好）。ResNet-32 和 MobileNetV2 在 CIFAR-10 上都用的是 `chenyaofo/pytorch-cifar-models` 第三方 checkpoint（`torch.hub.list('chenyaofo/pytorch-cifar-models')` 能看到全部可选架构），不用自己训练。

`run_experiments_tuned.py` 是独立的、超出论文字面复现范围的调优 pipeline（同预算下换排序方法/scale 计算方式，见其模块 docstring），`--ranking {greedy,surrogate,energy}` 三选一，宽通道模型（本文档坑 #13 提到的 MobileNetV2 那种）greedy 要记得加 `--channel-subsample 0` 避免默认值 32 在宽层上退化成随机排序：

```bash
python -m quant.run_experiments_tuned --experiment mobilenetv2_cifar10 --variant clip_p999 --ranking energy --max-batches 0
python -m quant.run_experiments_tuned --experiment mobilenetv2_cifar10 --variant clip_p999 --ranking greedy --channel-subsample 0 --max-batches 0
```
