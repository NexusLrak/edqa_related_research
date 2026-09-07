# eDQA 便携运行包

不依赖 Kaggle 或任何特定平台 API 的自包含压缩包——解压后在任意租借的 GPU 环境
（AutoDL、RunPod、Vast.ai、Lambda、Colab、公司自己的 GPU 服务器……）里都能跑，
不需要额外挂载数据集或访问外网下载权重。

## 目录结构

```
run.py                       通用入口脚本，命令行传模型名即可
requirements.txt
quant/                       项目的 quant 包（原样复制，未做任何修改）
vendor/pytorch_cifar_models/ ResNet-32 / MobileNetV2 的模型结构定义
                              （替代 torch.hub.load，不需要联网）
checkpoints/                 4 个模型各自的权重文件
data/                        cifar-10-python.tar.gz、tiny-imagenet-200.zip
output/                      运行后自动创建，rank 缓存 + 结果 json 会写到这里
run.ipynb                    只支持 notebook 的平台可以用这个（内部就是调 run.py）
```

## 快速开始

```bash
unzip edqa_portable_package.zip -d edqa
cd edqa
pip install -r requirements.txt   # 如果平台已经预装了匹配 CUDA 的 torch，可跳过这行

python run.py --experiment resnet18_tinyimagenet --ranking energy --max-batches 0
```

第一次针对某个 experiment 跑的时候，脚本会自动：
- 从 `data/cifar-10-python.tar.gz` 解压出 CIFAR-10（torchvision 自带逻辑，不会重新联网下载）
- 从 `data/tiny-imagenet-200.zip` 解压并重排 val 目录成 `ImageFolder` 需要的
  `val/<wnid>/*.JPEG` 布局（照搬项目里 `data/prepare_tinyimagenet_val.py` 的逻辑）

之后同一个 experiment 再跑就不会重复解压。

## 常用命令

```bash
# ViT 贪心 full-scan——本地预估~122小时跑不动才需要挪到租借平台，这是本包最初的用途
python run.py --experiment vit_b16_tinyimagenet --ranking greedy --channel-subsample 0 --max-batches 0

# MobileNetV2 贪心 full-scan（本地约84分钟/seed，量大但not infeasible，仍可以搬到更快的卡上跑）
python run.py --experiment mobilenetv2_cifar10 --ranking greedy --channel-subsample 0 --max-batches 0

# energy 排序（O(1)，一般1分钟内，任何模型都适用）
python run.py --experiment resnet18_tinyimagenet --ranking energy --max-batches 0
python run.py --experiment resnet32_cifar10 --ranking energy --max-batches 0
```

`--experiment` 可选：`resnet32_cifar10` / `mobilenetv2_cifar10` /
`resnet18_tinyimagenet` / `vit_b16_tinyimagenet`
`--ranking` 可选：`greedy` / `energy` / `surrogate` / `stratified` /
`saliency` / `fisher` / `fisher_a2` / `protection_gain` / `pg2` / `edqa_saliency`
（后六个见下方"关于`rank_gradient_surrogates.py`"）

**`--max-batches 0` 才是全量测试集**，默认值 8 只是本地快速迭代用的小样本，
准确率会明显偏高，不要拿默认值的数字去引用或对比（详见 `quant/RESULTS.md`
"8-batch subset 偏高" 那条记录）。

**`--channel-subsample 0` 才是论文原版、未做任何近似的 Algorithm 3**（每层
每个通道都实测），默认值 32 是速度上的简化，通道数宽的层（MobileNetV2/ViT）
下这个默认值可能是退化的（结果不再真正依赖准确率评估），具体见
`run.py` 里 `_get_ranks` 的注释。

## 先做小范围验证，别直接冲全量

新换一台没跑过的GPU机器，先别直接跑 `--channel-subsample 0`（可能是几十到上百小时的量级）。
用 `--layers-limit N` 只对前 N 层做真·full-scan 并计时，自动按这台机器实测的
"秒/候选"外推整个 layer 范围要多久，跑完直接退出，不会去跑 Table2/Figure3/Figure4
（只测速度，rank只覆盖了前几层不是有意义的准确率结果）：

```bash
# 只测第1层（比如 ViT 是 768 个通道的 self_attention），几分钟内出结果
python run.py --experiment vit_b16_tinyimagenet --ranking greedy --channel-subsample 0 --layers-limit 1

# 想再准一点，多测几层取平均
python run.py --experiment vit_b16_tinyimagenet --ranking greedy --channel-subsample 0 --layers-limit 3
```

输出长这样：

```
probe: 768 candidates in 42.3s (0.055s/candidate on this GPU)
extrapolated full scope (36 layers, 55296 candidates): 0.8 hours
```

拿到这个外推小时数再决定要不要跑完整 `--channel-subsample 0`（不加 `--layers-limit`）
的正式版本。跟 `quant/RESULTS.md` 里当初推算 ViT ~120小时不可行用的是同一个方法——
从一部分实测结果反推总量，不是拍脑袋。

## 适配别的租借平台需要改的地方

理论上不需要改任何东西——所有路径都是相对于 `run.py` 自身位置解析的
(`PKG_ROOT = os.path.dirname(os.path.abspath(__file__))`)，权重和
ResNet-32/MobileNetV2 的模型结构都打包在本地、不发起任何网络请求。
唯一需要注意的：

- **GPU/CUDA**：`requirements.txt` 里的 `torch`/`torchvision` 没有锁 CUDA
  版本——大多数租借平台会预装好匹配驱动的 torch，直接用平台自带的即可，
  不需要额外 `pip install`；如果平台是裸机/没预装，去
  https://pytorch.org/get-started/locally/ 选对应 CUDA 版本的安装命令。
- **磁盘空间**：解压后 checkpoints/ + data/ 加起来有小几个 GB，租借平台的
  持久化磁盘/临时磁盘空间要留够。
- **无网络限制的平台**（比如 Kaggle 关掉 internet 的 kernel）：本包已经
  针对这种情况设计——resnet32/mobilenetv2 不再用 `torch.hub.load` 从
  GitHub 拉取，而是用 `vendor/pytorch_cifar_models` 里打包好的模型结构 +
  `checkpoints/` 里的本地权重文件，全程不联网。

## 只支持 notebook 的平台

有些平台（Colab 等）习惯用 notebook 而不是脚本。`run.ipynb` 就是一个薄封装，
第一个 cell `pip install`，第二个 cell 改几个变量再调用 `run.run(...)`，
不需要额外写代码。

## 关于 `rank_gradient_surrogates.py`（论文核心贡献，2026-09-01 重建说明）

`saliency` / `fisher` / `fisher_a2` / `protection_gain` / `pg2` / `edqa_saliency`
（其中 `edqa_saliency` 即论文提出的 pg·|g|，是本项目的核心贡献）这六个排序准则，
实现代码在 `quant/rank_gradient_surrogates.py`。

**这份代码是重建出来的，不是原始代码**——产出论文这部分数字的原始探索脚本
（`pg_gradient_combos.py`、`nm_sweep_pg_gradient.py`、`boundary_n2_n4_mbv2_vit.py`
等）在打包提交前已经找不到了，`quant/`、`quant/diagnostics/`、`model/`、
`remote_backup/` 全部翻过，只剩下一份更早期、只测 ViT 的残片
（`model/kaggle_kernel/kaggle_vit_energy_saliency_blend.py`）。`rank_gradient_
surrogates.py` 是根据 `quant/RESULTS.md`/`quant/EXTENSION_WORK.md` 里记录的公式和
成本描述、以及那份残片里可用的 `retain_grad()`/`backward()` 写法重新实现的，
已经在 ResNet-32 参考配置 (n=3, m=3) 上跟已发表的数字做过验证（见下）。

**由此产生的实际影响**：重新跑这六个准则得到的数字，跟论文里已经写定的数字会有
seed/校准batch顺序层面的细微出入（GPU浮点不确定性），不保证逐位重现；单seed跑出
来的结果如果落在论文报告的多seed标准差量级附近（fisher/eDQA-saliency std约
0.2-0.4pp），可以认为重建是忠实的，但不应该拿新跑的数字直接替换论文正文——正文
数字仍以 `quant/RESULTS.md` 里记录的为准。

验证结果（ResNet-32/CIFAR-10, clip_p999, n=3 m=3, 单seed, 对照 RESULTS.md 记录，
2026-09-01 验证完成）：

| 准则 | 本次重建重跑 | RESULTS.md 记录 | 差值 |
|---|---|---|---|
| saliency | 88.32% | 88.42% | 0.10pp |
| fisher | 89.15% | 89.34%（3-seed, std 0.30pp） | 0.19pp |
| fisher_a2 | 88.37% | 88.29% | 0.08pp |
| protection_gain | 88.16% | 87.79% | 0.37pp |
| pg2 | 88.16%（与 protection_gain 逐位一致，符合"平方保序"预期） | 87.79% | 0.37pp |
| edqa_saliency | 89.09% | 89.26%（3-seed, std 0.24pp） | 0.17pp |

六项差值全部在 0.4pp 以内，与同量级准则的多-seed标准差（0.14-0.36pp）相符；
Direct baseline（80.61%）在六次独立跑里逐位一致，说明差异只来自排序阶段本身
的seed噪声，不是环境/实现层面的系统性偏差。另外做了一个独立于历史数字的一致性
检查：`rank_channels_via_pq(..., p=2, q=0)`（(p,q)家族里energy的位置）与仓库里
已有的 `rank_surrogates.rank_channels_via_energy` 在全部34层上给出逐层完全相同
的排序，零错配——两条独立实现互相印证。

**跨模型补充验证（2026-09-07）**：`edqa_saliency` 单独在 MobileNetV2、ViT-B/16
上补测了 $(n{=}3, m{=}1/2/3)$ 全部三点（脚本 `quant/verify_table_4_4.py`），
ViT 这次是重建代码第一次真正跑在 3D 张量（`encoder.layers.*`，channel_dim=2）
上，没有报错。$(3,3)$ 点跟已发表的3-seed均值对比：MobileNetV2 91.75% vs.
91.78%（差0.03pp），ViT 85.12% vs. 85.14%（差0.02pp）——同样落在噪声量级内。
$m{=}1,2$ 两个此前只能靠margin反推的点，现在也有了真实单seed测量：

| 模型 | m=1 | m=2 | m=3（对照3-seed均值） |
|---|---|---|---|
| MobileNetV2 | 85.30% | 90.86% | 91.75%（91.78%） |
| ViT-B/16 | 84.00% | 84.83% | 85.12%（85.14%） |

## 下载结果

跑完之后去 `output/` 目录里找：
- `ranks_*.json`——排序结果缓存，下次跑同样配置会直接复用，也可以下载回本地
  跟 `quant/ranks_*.json`（本地已有的结果）对比
- `results_{experiment}_{ranking}_{variant}_calibseed{N}.json`——Table2/Figure3/
  Figure4 的准确率数字 + 各阶段耗时，一次运行一份，可以直接下载引用
