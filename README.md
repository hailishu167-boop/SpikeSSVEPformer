# SpikeSSVEPformer：SNN 增强的 SSVEP 40 分类脑电解码

> 一个适合 BCI / 深度学习初学者学习的完整项目：从原始脑电数据预处理、频域特征提取、脉冲神经网络（SNN）模型设计，到训练、评估和可交互的「意念打字」Demo，全流程代码 + 详细中文教程。

基于 **SSVEPformer**（Chen et al., 2023, *Neural Networks*）的脉冲神经网络增强版本，在清华大学 **BETA 数据集**（35 名被试 / 40 类别 / 64 导联）上实现 SSVEP 脑电解码。

最佳模型（PLIF-SNN，1 秒时间窗）在留出被试 S35 上达到 **84.17% 准确率**、**ITR 154.19 bits/min**。

## 📚 这个项目能教你什么

如果你是刚接触脑机接口或深度学习的同学，按顺序读完本项目你会学到：

1. **SSVEP 脑电范式**的原理：为什么盯着闪烁的方块，大脑就会产生对应频率的信号 → [docs/01_ssvep_primer.md](docs/01_ssvep_primer.md)
2. **SNN（脉冲神经网络）**的核心概念：LIF/PLIF 神经元、脉冲发放率编码、替代梯度训练，全部用本科数学水平讲解 → [docs/02_snn_explained.md](docs/02_snn_explained.md)
3. **一个真实科研项目的完整工程流程**：数据预处理 → 缓存 → 模型 → 训练 → 评估 → Demo，逐文件讲解 → [docs/03_code_walkthrough.md](docs/03_code_walkthrough.md)
4. **调参和实验思维**：数据增强、早停、余弦退火、固定拆分 vs LOSO 交叉验证的区别

## 🧠 模型架构

```
原始时域 EEG（1 秒, 11 个枕叶导联）  (B, 11, 250)
    ↓  FFT（nfft=1000, 频率分辨率 0.25 Hz）
复数频谱
    ↓  Filter-Bank：3 个子带 (8-45 / 16-45 / 24-45 Hz)，含可学习子带权重
每个子带独立进入一个子网络 Subnet：
    ├─ ChComb        通道组合（1×1 Conv + LayerNorm + GELU）
    ├─ PLIFEncoder   CNN + 残差 + PLIF 脉冲编码（tau 可学习）
    └─ PLIFEncoder   同上，再叠一层
    ↓  3 个子带特征
PLIFFeatureFusion   PLIF 发放率编码 + softmax 子带加权融合
    ↓
MlpHead-LIF         Linear → LIF「脉冲瓶颈」→ Linear → 40 类
```

同文件中还提供**纯 ANN 基线** `FBSSVEPformerV4PLIFBaseline`（把所有 LIF/PLIF 换成普通激活），方便你做消融对比实验。

## 📁 项目结构

| 路径 | 说明 |
|------|------|
| `scripts/preprocess.py` | 数据预处理：`.mat` → 带通滤波 → 缓存 `.npy`（生成 `cache_1s/`、`cache_5s/`） |
| `scripts/dataset.py` | PyTorch Dataset：按被试加载缓存数据 |
| `scripts/analyze_per_class.py` | 逐类别准确率分析，找出模型「认不出」的弱类别并生成类别权重 |
| `models/plif_ssvepformer.py` | ⭐ 主模型：PLIF 神经元 SNN-SSVEPformer（另含 ANN 基线） |
| `models/deep_temporal_snn.py` | 变体：更深层时序 SNN |
| `models/filterbank_ssvepformer.py` | 变体：5 子带 Filter-Bank |
| `train/train_plif.py` | 主模型训练（固定拆分 + 弱类别 3× 数据增强 + 余弦退火 + 早停） |
| `train/train_deep_temporal.py` | 深层时序 SNN 训练脚本 |
| `train/train_filterbank.py` | Filter-Bank 版本训练脚本 |
| `evaluate.py` | 评估脚本：逐被试准确率、混淆矩阵绘图 |
| `demo/typing_demo.py` | 🎮 交互式「意念打字」Demo（pygame 可视化） |
| `checkpoints/` | 训练好的模型权重（`.pth`） |
| `data_info/` | BETA 数据集说明、64 导联位置文件、40 个刺激频率/相位表 |
| `results_plif/`、`results_deep_temporal/` | 训练曲线（PNG）与指标（JSON） |
| `docs/` | 📖 三篇中文教程（见上文） |

## 🚀 快速开始

### 1. 安装环境

```bash
pip install -r requirements.txt
```

依赖：PyTorch、numpy、scipy、scikit-learn、matplotlib、tqdm、[spikingjelly](https://github.com/fangwei123456/spikingjelly)、pygame（Demo 用）。

### 2. 下载数据

本项目使用清华大学 **BETA SSVEP Benchmark**（学术研究免费，但需申请，不允许二次分发）：

1. 到官方页面申请下载：http://bci.med.tsinghua.edu.cn/download.html
2. 把 `S1.mat` ~ `S35.mat` 放到仓库根目录（这些文件已在 `.gitignore` 中排除）

### 3. 预处理（生成缓存，跑一次以后就不用再跑）

```bash
python scripts/preprocess.py
```

### 4. 训练与评估

```bash
# 训练主模型（S1-S34 训练 / S35 测试，GPU 上约几十分钟）
python train/train_plif.py

# 两个结构变体
python train/train_deep_temporal.py
python train/train_filterbank.py

# 评估
python evaluate.py
```

### 5. 最好玩的部分：意念打字 Demo

```bash
python demo/typing_demo.py
```

你在物理键盘上打字，程序取出被试 S35 注视对应字符时的真实 EEG，让模型逐字符「读心」预测你按的是哪个键，实时对比、统计准确率——直观感受 84% 准确率到底意味着什么。

## 📊 实验结果

固定拆分（S1–S34 训练，S35 测试），1 秒时间窗：

| 模型 | 参数量 | 最佳准确率 | ITR |
|------|--------|-----------|-----|
| **PLIF-SNN（主模型）** | 1.74 M | **84.17%** | **154.19 bits/min** |
| Deep Temporal SNN | 1.88 M | 83.75% | 152.91 bits/min |

训练策略：Adam + 余弦退火（lr=1e-3）、dropout=0.3、`T_snn=12`、弱类别（A / F / 4）3× 过采样增强、early stopping（patience=80）。

参考（论文数值，BETA 40 类 inter-subject，1 秒窗）：SSVEPformer 80.40%、FB-SSVEPformer 83.19%。

> ⚠️ **学术诚信提示**：本项目的结果是「留出单个被试 S35」的固定拆分，不是严格的 35 折 LOSO（留一被试交叉验证），与论文数值不能直接对比。LOSO 的评估代码在 `evaluate.py` 里已经写好，留作练习。详见 [docs/03_code_walkthrough.md](docs/03_code_walkthrough.md) 的「局限与改进方向」。

## 🗺️ 推荐学习路线

| 阶段 | 内容 | 产出 |
|------|------|------|
| 第 1 天 | 读 [docs/01](docs/01_ssvep_primer.md)，搞懂 SSVEP 范式和 BETA 数据集 | 能向同学讲清楚「频率 → 标签」的原理 |
| 第 2 天 | 读 [docs/02](docs/02_snn_explained.md)，搞懂 LIF/PLIF 和替代梯度 | 能手推 LIF 膜电位递推公式 |
| 第 3 天 | 按 [docs/03](docs/03_code_walkthrough.md) 跑通预处理 + 训练 | 复现 84% 准确率 |
| 第 4 天 | 跑 Demo、改参数做消融（T_snn、子带数、去掉 SNN 层） | 一张自己的对比实验表 |
| 之后 | 挑战 docs/03 末尾的「改进方向」 | 你自己的新模型 |

## 📖 引用

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

MIT（代码）。BETA 数据集本身有自己的使用协议，请勿二次分发数据。
