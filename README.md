# SpikeSSVEPformer: SNN 增强的 SSVEP 40 分类解码

基于 **SSVEPformer** (Chen et al., 2023, *Neural Networks*) 的脉冲神经网络（SNN）增强版本，在 **BETA 数据集**（35 subjects / 40 classes / 64 channels）上实现 SSVEP 脑电解码，并附带一个可交互的「意念打字」演示程序。

最佳模型（V5-PLIF, 1s 时间窗）在留出被试 S35 上达到 **84.17% 准确率**，**ITR 154.19 bits/min**。

## 核心架构

```
原始时域 EEG (B, C, T)
    ↓
FFT 频域变换 → 复数谱（实部 + 虚部）
    ↓
Channel Combination (Conv1d)
    ↓
Encoder 1 (CNN + MLP + Residual)
    ↓
SNN 层（PLIF/LIF 神经元, spikingjelly）  ← SNN 增强
    ↓
Encoder 2 (CNN + MLP + Residual)
    ↓
MLP Classification Head → 40 classes
```

V7 在此基础上进一步引入 **5 子带 Filter-Bank** 结构。

## 项目结构

| 路径 | 说明 |
|------|------|
| `scripts/preprocess.py` | 数据预处理：`.mat` → 滤波缓存 `.npy`（生成 `cache_1s/`、`cache_5s/`） |
| `scripts/dataset.py` | 数据集加载与 DataLoader |
| `scripts/analyze_per_class.py` | 逐类别准确率分析，输出弱类别权重 |
| `models/plif_ssvepformer.py` | 主模型：PLIF 神经元 SNN-SSVEPformer（最优） |
| `models/deep_temporal_snn.py` | 变体：深层时序 SNN |
| `models/filterbank_ssvepformer.py` | 变体：5 子带 Filter-Bank |
| `train/train_plif.py` | 主模型训练脚本（固定拆分 + 弱类别 3× 数据增强） |
| `train/train_deep_temporal.py` | 深层时序 SNN 训练脚本 |
| `train/train_filterbank.py` | Filter-Bank 版本训练脚本 |
| `evaluate.py` | 模型评估（支持逐被试准确率与混淆矩阵绘图） |
| `demo/typing_demo.py` | 交互式「意念打字」Demo（pygame 可视化） |
| `checkpoints/` | 训练好的模型权重（`.pth`） |
| `data_info/` | BETA 数据集说明、64 导联位置、40 个刺激频率/相位表 |
| `results_plif/`、`results_deep_temporal/` | 训练曲线（PNG）与指标（JSON） |

## 环境安装

```bash
pip install -r requirements.txt
```

依赖：PyTorch、numpy、scipy、scikit-learn、matplotlib、tqdm、[spikingjelly](https://github.com/fangwei123456/spikingjelly)、pygame（Demo 用）。

## 数据准备

本项目使用清华大学 **BETA SSVEP Benchmark**：

1. 到官方页面申请并下载：http://bci.med.tsinghua.edu.cn/download.html
2. 将 `S1.mat` ~ `S35.mat` 放到仓库根目录（`.mat` 文件已在 `.gitignore` 中排除，不会被上传）
3. 运行预处理生成缓存：

```bash
python scripts/preprocess.py
```

## 训练与评估

```bash
# 训练主模型（PLIF-SNN，S1-S34 训练 / S35 测试）
python train/train_plif.py

# 两个变体
python train/train_deep_temporal.py
python train/train_filterbank.py

# 评估
python evaluate.py
```

## 交互式 Demo

```bash
python demo/typing_demo.py
```

在物理键盘上打字，模型用 S35 的真实 EEG trial 逐字符「读心」预测你按下的字符，实时对比目标文本与预测文本，并统计打字速度与准确率。

## 实验结果

固定拆分（S1–S34 训练，S35 测试），1s 时间窗：

| 模型 | 参数量 | 最佳准确率 | ITR |
|------|--------|-----------|-----|
| **V5: PLIF-SNN (SpikeSSVEPformer)** | 1.74 M | **84.17%** | **154.19 bits/min** |
| V6: Deep Temporal SNN | 1.88 M | 83.75% | 152.91 bits/min |

训练策略：Adam + 余弦退火（lr=1e-3）、dropout=0.3、`T_snn=12`、弱类别（A / F / 4）3× 数据增强、early stopping（patience=80）。

参考（论文数值，BETA 40-class inter-subject）：SSVEPformer 80.40%，FB-SSVEPformer 83.19%（1s 窗）。

## 设计思路

**为什么用频域输入？** SSVEP 的核心判别特征是频率与谐波。40 个刺激频率间隔仅 0.2 Hz（8.0–15.8 Hz），时域 CNN 难以直接区分；FFT 复数谱将频率信息显式编码，模型可直接在频率维度学习。

**为什么用 SNN？** SSVEP 是大脑的周期性电响应，LIF/PLIF 神经元的脉冲动力学天然适合建模这种周期性，同时引入时序动态增强频域特征。SNN 层使用 spikingjelly 实现，替代梯度函数为 ATan。

## 引用

```bibtex
@article{chen2023transformer,
  title={A transformer-based deep neural network model for SSVEP classification},
  author={Chen, Jianbo and Zhang, Yangsong and Pan, Yudong and Xu, Peng and Guan, Cuntai},
  journal={Neural Networks},
  volume={164},
  pages={521--534},
  year={2023}
}

@article{liu2020beta,
  title={BETA: A large benchmark database toward SSVEP-BCI application},
  author={Liu, Baolin and Huang, Xiaoshan and Wang, Yijun and Chen, Xiaogang and Gao, Xiaorong},
  journal={Frontiers in Neuroscience},
  volume={14},
  pages={627},
  year={2020}
}

@article{fang2023spikingjelly,
  title={SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
  author={Fang, Wei and others},
  journal={Science Advances},
  volume={9},
  number={40},
  pages={eadi1480},
  year={2023}
}
```

## 许可证

MIT
