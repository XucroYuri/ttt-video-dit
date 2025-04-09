# TTT-Video 在 RTX 4090 24G 显卡集群上的优化方案

本文档提供了在 RTX 4090 24G 显卡集群上运行 TTT-Video 项目的全面优化方案，从硬件环境配置、训练集准备到代码修改三个维度进行详细说明。

## 1. 硬件环境配置

### 1.1 CUDA 和驱动要求

- **CUDA 版本**: 使用 CUDA 12.3+ (项目要求)
- **驱动版本**: 使用与 CUDA 12.3+ 兼容的最新 NVIDIA 驱动
- **GCC 版本**: 确保 GCC 11+ 可用，用于编译 TTT-MLP 内核

### 1.2 SLURM 配置调整

对于使用 SLURM 的集群环境，修改 `scripts/train_submitit.sh` 脚本：

```bash
# 修改 GPU 类型指定
--constraint="gpu_model=RTX_4090" 

# 调整内存限制
--mem=64G 

# 减少每节点 GPU 数量
--ntasks-per-node=4  # 根据实际节点上的 4090 数量调整
```

### 1.3 单节点训练脚本调整

修改 `scripts/train_singlenode.sh` 脚本：

```bash
# 调整为实际可用的 GPU 数量
NUM_GPUS=4  # 根据实际节点上的 4090 数量调整

# 增加环境变量以优化内存使用
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

## 2. 训练集准备优化

### 2.1 视频预处理优化

修改 `data/precomp_video.py` 中的视频预处理参数：

- **降低视频分辨率**：将输入视频处理为较低分辨率，建议 720p 或更低
- **降低帧率**：从默认的 16fps 降低到 8fps，减少处理帧数
- **批处理大小调整**：将预处理批处理大小从默认值降低到 1-2

```python
# 在 precompute_episode 函数中修改批处理大小
batch_size = 1  # 降低批处理大小以减少内存使用
```

### 2.2 数据加载优化

修改 `ttt/datasets/preembedding_dataset.py` 中的数据加载参数：

```python
# 在 create_dataloader 方法中
num_workers = 2  # 减少工作线程数
pin_memory = False  # 对于内存受限的情况可以禁用
```

### 2.3 预计算嵌入存储优化

- 对于长视频（9秒以上），考虑降低采样率或分辨率以减少嵌入大小
- 使用压缩格式存储预计算的嵌入，如使用 `torch.save(..., _use_new_zipfile_serialization=True)`

## 3. 代码修改

### 3.1 批处理大小调整

修改训练配置文件 `configs/train/ttt-mlp/3s.toml`：

```toml
[model]
mini_batch_size = 16  # 从 64 降低到 16

[training]
global_batch_size = 16  # 从 64 降低到 16
grad_accum_steps = 4  # 增加梯度累积步数以保持有效批量大小
```

### 3.2 并行策略调整

修改并行策略参数以适应 4090 显卡：

```toml
[parallelism]
fsdp_unsharded_dtype = 'bfloat16'  # 保持 bfloat16 以节省内存
dp_replicate = 1  # 减少复制
dp_sharding = 4  # 增加分片以减少每个 GPU 的内存使用
tp_sharding = 1  # 对于 3 秒视频可以不使用张量并行
```

对于更长的视频（9秒以上），建议：

```toml
[parallelism]
dp_replicate = 1
dp_sharding = 2
tp_sharding = 2  # 启用张量并行以分散内存使用
```

### 3.3 梯度检查点频率调整

增加梯度检查点频率以减少内存使用：

```toml
[remat]
transformer_checkpoint_layer_group_size = 1  # 每层都使用检查点
scan_checkpoint_group_size = 8  # 从 16 减少到 8，增加检查点频率
```

### 3.4 模型大小调整

对于 5B 参数模型，考虑减少 Transformer 层数：

```python
# 在 ttt/models/configs.py 中添加一个适合 4090 的配置
"4090-friendly": {
    "model_dim": 2048,  # 从 3072 减少到 2048
    "num_heads": 32,    # 从 48 减少到 32
    "num_layers": 32,   # 从 42 减少到 32
    "text_dim": 4096,
},
```

然后在配置文件中使用这个新的模型大小：

```toml
[model]
size = "4090-friendly"  # 使用新的模型配置
```

### 3.5 混合精度训练优化

确保启用混合精度训练，并考虑使用 float16 而不是 bfloat16（如果 4090 上 float16 性能更好）：

```toml
[parallelism]
fsdp_unsharded_dtype = 'float16'  # 可选：如果 float16 在 4090 上性能更好
```

## 4. 运行建议

### 4.1 分阶段训练

- 从最短的视频长度（3秒）开始训练
- 成功后再尝试更长的视频长度
- 对于每个阶段，先使用小批量进行测试运行，确认内存使用情况

### 4.2 监控与调试

- 使用 `nvidia-smi` 监控 GPU 内存使用情况
- 启用 PyTorch 内存分析器进行内存使用分析：
  ```python
  from torch.profiler import profile, record_function, ProfilerActivity
  ```
- 在训练脚本中添加内存使用日志

### 4.3 容错与恢复

- 增加检查点保存频率：
  ```toml
  [checkpoint]
  interval = 100  # 从 500 减少到 100
  ```
- 启用自动恢复功能，确保训练中断后可以继续

## 总结

通过以上优化，TTT-Video 项目应该能够在 RTX 4090 24G 显卡集群上稳定运行。关键点是减小批处理大小、优化并行策略、增加梯度检查点频率，以及根据需要调整模型大小。对于更长的视频，可能需要进一步减小批处理大小或增加张量并行度。