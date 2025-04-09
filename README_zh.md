# TTT-Video
<img src="./docs/figures/hero.png" alt="Hero" style="width:100%;"/>
TTT-Video 是一个用于微调扩散变换器(diffusion transformers)进行风格迁移和上下文扩展的代码库。我们使用测试时训练(Test-Time Training, TTT)层处理全局上下文中的长程关系，同时重用原始预训练模型的注意力层对每个三秒片段进行局部注意力处理。<br> <br>
在这个代码库中，我们包含了长达63秒视频生成的训练和推理代码。我们首先在原始预训练的3秒视频长度上微调模型，用于风格迁移和整合TTT层。然后，我们分阶段在9秒、18秒、30秒和63秒的视频长度上进行训练，以实现上下文扩展。

[中文文档](./README_zh.md) | [English](./README.md)

## 模型架构
![架构图](./docs/figures/integration.jpg)

我们的架构改进了[CogVideoX](https://github.com/THUDM/CogVideo) 5B模型（一个用于文本到视频生成的扩散变换器），通过整合TTT层。原始预训练的注意力层被保留用于对每个3秒片段及其对应文本的局部注意力处理。此外，TTT层被插入以处理全局序列及其反转版本，其输出通过残差连接进行门控。

为了将上下文扩展到预训练的3秒片段之外，每个片段都与文本和视频嵌入交错排列。

有关我们架构的更详细解释，请参阅我们的[论文](https://test-time-training.github.io/video-dit/assets/ttt_cvpr_2025.pdf)。

## 环境配置

### 依赖项
您可以使用conda（推荐）或虚拟环境安装此项目所需的依赖项。

#### Conda
```bash
conda env create -f environment.yaml
conda activate ttt-video
```

#### Pip
```bash
pip install -e .
```

### 内核安装
安装依赖项后，您必须安装TTT-MLP内核。

```bash
git submodule update --init --recursive
(cd ttt-tk && python setup.py install)
```

> **注意**：您必须安装cuda toolkit（12.3+）和gcc11+才能构建TTT-MLP内核。我们仅支持在H100上训练TTT-MLP。您可以在[这里](https://anaconda.org/nvidia/cuda-toolkit)安装cuda toolkit。

### CogVideoX预训练模型
请按照[这里](https://github.com/THUDM/CogVideo/blob/main/sat/README.md)的说明获取VAE和T5编码器。要获取预训练权重，请从[HuggingFace](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/transformer)下载`diffusion_pytorch_model-00001-of-00002.safetensors`和`diffusion_pytorch_model-00002-of-00002.safetensors`文件。

> **注意**：我们仅使用5B权重，而非2B权重。请确保您下载的是正确的模型。

## 注释下载
训练期间使用的3秒Tom and Jerry片段的文本注释可以在[这里](https://test-time-training.github.io/video-dit/assets/annotations.zip)访问。

## 其他文档
- [数据集](./docs/dataset.md)
- [训练](./docs/training.md)
- [采样](./docs/sampling.md)

## RTX 4090优化分支更新日志

我们创建了这个优化分支，专门针对RTX 4090 24G显卡进行了一系列优化，使模型能够在消费级GPU上高效训练。以下是主要技术改进：

### 2025年4月9日 - 配置优化
### 2025年4月9日 - RTX 4090综合优化
- 创建RTX 4090专用配置文件，以3秒视频长度为主，最大支持到10秒，并提供自动化优化脚本
- 通过改进批处理大小、梯度累积和检查点机制优化内存使用
- 采用FP16混合精度训练和针对Ampere架构优化的TTT-MLP内核提升计算性能
- 引入自适应梯度缩放和动态检查点策略提高训练稳定性
- 开发综合性能监控系统，提供实时内存可视化和自适应资源管理

### 视频长度优化策略

为保持与原项目的训练集特性一致，我们采取了谨慎的视频长度优化策略：

- **3秒视频（推荐）**：与原项目完全一致，确保训练效果和模型行为与原始实现相同
- **5秒视频（可选）**：适度扩展，在保持原项目核心特性的同时提供略长的上下文
- **10秒视频（实验性）**：最大支持长度，仅用于实验目的，不保证与原项目有相同的效果

我们不建议尝试更长的视频长度（如18秒、30秒或63秒），以避免训练集与原项目出现过大差异，这可能导致模型行为发生显著变化。

### 技术优化细节

针对不同视频长度，我们采用了以下技术优化措施：

#### 3秒视频优化（基准配置）
- **内存管理**：批处理大小从64减少到16，梯度累积步数增加到4
- **计算加速**：采用FP16混合精度训练，检查点组大小设为8
- **并行策略**：保持单GPU张量并行，优化数据并行分片

#### 5秒视频优化
- **内存管理**：批处理大小进一步减少到12，梯度累积步数增加到6
- **计算加速**：检查点组大小设为6，保持计算与内存使用的平衡
- **稳定性优化**：增加检查点保存频率，确保训练过程可恢复

#### 10秒视频优化
- **内存管理**：批处理大小减少到8，梯度累积步数增加到8
- **计算加速**：检查点组大小设为4，在多GPU环境下启用张量并行
- **稳定性优化**：引入自适应梯度缩放，提高数值稳定性

### 使用方法

要应用RTX 4090优化设置，只需运行以下命令：

```bash
bash scripts/optimize_for_4090.sh [视频长度(3s/5s/10s)]
```

这将自动创建优化后的配置文件和训练脚本。我们建议优先使用3秒视频长度，这与原项目保持一致，可以确保训练效果。对于5秒和10秒的视频长度，可以尝试但不保证与原项目有相同的效果。不建议使用更长的视频长度，以避免与原项目产生过大差异。

生成脚本后，您可以使用以下命令开始训练：

```bash
bash scripts/train_4090_3s.sh
```

有关详细的优化分析和技术决策，请参阅以下文档：
- [4090优化方案技术分析](./docs/4090_optimization_analysis.md)
- [4090优化指南](./docs/4090_optimization_guide.md)
- [优化分支开发记录](./docs/optimization_branch_notes.md)