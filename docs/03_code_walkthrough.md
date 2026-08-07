# 03 · 代码导读与复现教程

> 按数据流向走完整个项目。

## 0. 全局数据流

```
S1.mat ~ S35.mat（原始数据，每个约 100 MB）
   │  scripts/preprocess.py：选导联 → 带通滤波 → 切片 → .npy
   ▼
cache_1s/ 或 cache_5s/（S1_data.npy, S1_labels.npy, ...）
   │  train/train_plif.py：加载缓存 → 数据增强 → 训练
   ▼
results_plif/（模型权重 .pth + 指标 .json + 曲线 .png）
   │  evaluate.py / demo/typing_demo.py
   ▼
评估报告 / 意念打字 Demo
```

## 1. `scripts/preprocess.py` —— 数据预处理

**做的事**：把 BETA 的 `.mat` 原始文件变成训练可直接加载的 `.npy` 缓存。

关键步骤：

1. **选导联**：64 导联里只保留 11 个枕叶/顶枕导联（`PZ, PO5, PO3, POZ, PO4, PO6, PO8, O1, OZ, O2, CB1, CB2` 中的 11 个）。理由：SSVEP 源于视觉皮层，枕叶导联信号最强；丢掉其他导联既去噪又降维。
2. **带通滤波**：4 阶 Butterworth，通带 5–50 Hz（`butter` + `filtfilt`）。`filtfilt` 是零相位滤波（正反向各滤一次），不会让波形产生时间偏移——EEG 处理的标准操作。
3. **切片**：每个 trial 从刺激开始后 0.14 s 处截取（0.14 s 是视觉诱发电位的生理延迟，BETA 论文给出的数值），再取 1 s 或 5 s 窗口。

输出：`cache_1s/S{n}_data.npy`（shape: trials×11×250）和 `S{n}_labels.npy`。



## 2. `models/plif_ssvepformer.py` —— 主模型

按 `forward` 的顺序读（约 261–279 行）：

```python
x_fft = torch.fft.fft(x, n=self.nfft) / x.shape[-1]   # ① FFT 到频域
x_subbands = self.filter_bank(x_fft)                   # ② 切成 3 个子带
# ③ 每个子带：取实部+虚部拼接 → 独立子网络（ChComb + PLIFEncoder×2）
# ④ PLIFFeatureFusion：softmax 加权融合 3 个子带特征
# ⑤ MlpHeadLIF：分类
```

各模块的细节（含 LIF/PLIF 数学）见 [02_snn_explained.md](02_snn_explained.md)，这里只补两个容易忽略的设计：

- **为什么保留复数的实部+虚部？** 频谱的幅度告诉你「这个频率有多强」，相位告诉你「响应的时延」。BETA 的 40 个刺激各有特定初始相位，相位也是判别信息，所以不能像很多谱分析方法那样只取幅度谱。
- **Filter-Bank 子带 (8-45, 16-45, 24-45 Hz)**：基波主要在低频子带，二次谐波落在 16 Hz 以上，三次在 24 Hz 以上。分子带让模型显式分离基波/谐波信息，再用可学习权重融合——这个技巧来自 FB-SSVEPformer，是 SSVEP 领域的经典设计（最早可追溯到 FBCCA）。



## 3. `train/train_plif.py` —— 训练脚本

工程细节：

| 技巧 | 位置 | 作用 |
|------|------|------|
| 固定拆分 S1–S34 / S35 | `run_fixed_split` | 跨被试评估，快速迭代 |
| 弱类别 3× 过采样 | 数据增强部分 | 先跑 `scripts/analyze_per_class.py` 找出弱类别（A/F/4），再多复制几遍 |
| 余弦退火学习率 | `cos, pi` 相关 | lr 从 1e-3 平滑降到 0，后期收敛更稳 |
| Early stopping | `patience=80` | 测试准确率 80 轮不涨就停，保留历史最佳权重 |
| 保存最佳模型 | `v4_plif_best_model.pth` | 永远保存验证集最好的那次，而不是最后一次 |

**复现主结果**（GPU 上约几十分钟）：

```bash
python scripts/preprocess.py        # 只需一次
python train/train_plif.py
```

结束后看 `results_plif/v4_plif_results.json`，应该得到约 80%左右 的准确率。

## 4. `evaluate.py` 与 `scripts/analyze_per_class.py`

- `evaluate.py`：加载训练好的权重做系统评估，`--plot` 生成混淆矩阵和逐被试柱状图。
- `analyze_per_class.py`：逐类别准确率分析。它告诉你模型在哪些字符上犯错最多，并输出 `class_weights.json`——主模型的「弱类别增强」就是这么来的。**先分析弱点，再针对性增强**，是 ML 工程里非常值得学的思路。

## 5. `demo/typing_demo.py` —— 意念打字

pygame 写的可视化 Demo。流程：你按一个物理键 → 程序从 `cache_1s` 取 S35 注视该字符的一个真实 trial → 模型预测 → 屏幕对比显示。注意它只是「回放真实数据」，不是实时采集——真正的实时系统还需要放大器硬件和滑窗推理，但算法核心完全一样。

## 6. 常见坑（FAQ）

1. **`RuntimeError: ... spikingjelly reset`**：每个 forward 前必须 reset 神经元状态，检查是否漏了 `functional.reset_net()` 或 `lif.reset()`。
2. **准确率先涨后崩**：学习率过大或 dropout 太小；主模型用 lr=1e-3 + dropout=0.3 是调过的。
3. **Windows 中文路径**：`loadmat` / `np.load` 对中文路径一般没问题，但如果用 MATLAB 引擎或某些 C 扩展可能报错，建议数据路径用纯英文。
4. **结果复现不完全一致**：GPU 上的浮点非确定性 + 数据增强随机性，±1% 波动正常。想严格复现就固定 `torch.manual_seed`。



## 7. 推荐拓展阅读

- SSVEPformer 原论文：Chen et al., *Neural Networks*, 2023
- BETA 数据集论文：Liu et al., *Frontiers in Neuroscience*, 2020
- SpikingJelly 官方教程（中文，非常友好）：https://spikingjelly.readthedocs.io
- 经典无训练方法：FBCCA（Lin et al., 2006）
