#!/bin/bash

# 此脚本用于自动应用RTX 4090 24G显卡优化配置
# 使用方法: bash scripts/optimize_for_4090.sh [视频长度(3s/5s/10s)]

set -e

VIDEO_LENGTH=${1:-"3s"}

# 验证视频长度参数
if [ "${VIDEO_LENGTH}" != "3s" ] && [ "${VIDEO_LENGTH}" != "5s" ] && [ "${VIDEO_LENGTH}" != "10s" ]; then
    echo "错误: 视频长度必须是 3s、5s 或 10s 之一"
    echo "用法: bash scripts/optimize_for_4090.sh [视频长度(3s/5s/10s)]"
    exit 1
fi

echo "正在为 ${VIDEO_LENGTH} 视频长度配置 RTX 4090 优化设置..."
echo "注意: 建议优先使用3秒视频长度，这与原项目保持一致，可以确保训练效果。"

# 确定可用GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 ${NUM_GPUS} 个GPU"

# 创建优化后的配置文件目录
OPTIMIZED_CONFIG_DIR="./configs/train/ttt-mlp-4090"
mkdir -p "${OPTIMIZED_CONFIG_DIR}"

# 根据视频长度选择源配置文件
SOURCE_CONFIG="./configs/train/ttt-mlp/${VIDEO_LENGTH}.toml"
if [ ! -f "${SOURCE_CONFIG}" ]; then
    echo "错误: 配置文件 ${SOURCE_CONFIG} 不存在"
    exit 1
fi

# 创建优化后的配置文件
OPTIMIZED_CONFIG="${OPTIMIZED_CONFIG_DIR}/${VIDEO_LENGTH}.toml"
cp "${SOURCE_CONFIG}" "${OPTIMIZED_CONFIG}"

# 应用优化配置
if [ "${VIDEO_LENGTH}" == "3s" ]; then
    # 3秒视频优化配置 - 与原项目保持最大兼容性
    sed -i 's/mini_batch_size = 64/mini_batch_size = 16/' "${OPTIMIZED_CONFIG}"
    sed -i 's/global_batch_size = 64/global_batch_size = 16/' "${OPTIMIZED_CONFIG}"
    sed -i 's/grad_accum_steps = 1/grad_accum_steps = 4/' "${OPTIMIZED_CONFIG}"
    sed -i 's/dp_replicate = 8/dp_replicate = 1/' "${OPTIMIZED_CONFIG}"
    sed -i "s/dp_sharding = 8/dp_sharding = ${NUM_GPUS}/" "${OPTIMIZED_CONFIG}"
    sed -i 's/tp_sharding = 1/tp_sharding = 1/' "${OPTIMIZED_CONFIG}"
    sed -i 's/scan_checkpoint_group_size = 16/scan_checkpoint_group_size = 8/' "${OPTIMIZED_CONFIG}"
    sed -i 's/interval = 500/interval = 100/' "${OPTIMIZED_CONFIG}"
elif [ "${VIDEO_LENGTH}" == "5s" ]; then
    # 5秒视频优化配置 - 适度扩展但保持与原项目相似性
    sed -i 's/mini_batch_size = 64/mini_batch_size = 12/' "${OPTIMIZED_CONFIG}"
    sed -i 's/global_batch_size = 64/global_batch_size = 12/' "${OPTIMIZED_CONFIG}"
    sed -i 's/grad_accum_steps = 1/grad_accum_steps = 6/' "${OPTIMIZED_CONFIG}"
    sed -i 's/dp_replicate = 8/dp_replicate = 1/' "${OPTIMIZED_CONFIG}"
    sed -i "s/dp_sharding = 8/dp_sharding = ${NUM_GPUS}/" "${OPTIMIZED_CONFIG}"
    sed -i 's/tp_sharding = 1/tp_sharding = 1/' "${OPTIMIZED_CONFIG}"
    sed -i 's/scan_checkpoint_group_size = 16/scan_checkpoint_group_size = 6/' "${OPTIMIZED_CONFIG}"
    sed -i 's/interval = 500/interval = 75/' "${OPTIMIZED_CONFIG}"
else
    # 10秒视频优化配置 - 最大支持长度，谨慎配置
    sed -i 's/mini_batch_size = 64/mini_batch_size = 8/' "${OPTIMIZED_CONFIG}"
    sed -i 's/global_batch_size = 64/global_batch_size = 8/' "${OPTIMIZED_CONFIG}"
    sed -i 's/grad_accum_steps = 1/grad_accum_steps = 8/' "${OPTIMIZED_CONFIG}"
    sed -i 's/dp_replicate = 8/dp_replicate = 1/' "${OPTIMIZED_CONFIG}"
    
    # 计算dp_sharding和tp_sharding
    if [ ${NUM_GPUS} -ge 4 ]; then
        DP_SHARDING=2
        TP_SHARDING=2
    else
        DP_SHARDING=${NUM_GPUS}
        TP_SHARDING=1
    fi
    
    sed -i "s/dp_sharding = 8/dp_sharding = ${DP_SHARDING}/" "${OPTIMIZED_CONFIG}"
    sed -i "s/tp_sharding = 1/tp_sharding = ${TP_SHARDING}/" "${OPTIMIZED_CONFIG}"
    sed -i 's/scan_checkpoint_group_size = 16/scan_checkpoint_group_size = 4/' "${OPTIMIZED_CONFIG}"
    sed -i 's/interval = 500/interval = 50/' "${OPTIMIZED_CONFIG}"
fi

# 创建优化后的训练脚本
OPTIMIZED_SCRIPT="./scripts/train_4090_${VIDEO_LENGTH}.sh"
cp "./scripts/train_singlenode.sh" "${OPTIMIZED_SCRIPT}"

# 修改训练脚本
sed -i "s/NUM_GPUS=8/NUM_GPUS=${NUM_GPUS}/" "${OPTIMIZED_SCRIPT}"
sed -i "s|CONFIG_FILE=\"./configs/train/ttt-mlp/3s.toml\"|CONFIG_FILE=\"${OPTIMIZED_CONFIG}\"|" "${OPTIMIZED_SCRIPT}"
sed -i "s/EXP_NAME=\".*\"/EXP_NAME=\"4090-ttt-video-${VIDEO_LENGTH}-optimized\"/" "${OPTIMIZED_SCRIPT}"

# 添加内存优化环境变量
sed -i '/conda activate ttt-video/a\\nexport PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"' "${OPTIMIZED_SCRIPT}"

# 添加视频预处理优化脚本
PRECOMP_SCRIPT="./scripts/precompute_4090.sh"
cat > "${PRECOMP_SCRIPT}" << 'EOF'
#!/bin/bash

# 此脚本用于在RTX 4090上优化视频预处理
# 使用方法: bash scripts/precompute_4090.sh [输入视频目录] [输出目录] [VAE权重路径]

INPUT_DIR=$1
OUTPUT_DIR=$2
VAE_WEIGHTS=$3

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$VAE_WEIGHTS" ]; then
    echo "用法: $0 [输入视频目录] [输出目录] [VAE权重路径]"
    exit 1
fi

# 设置优化参数
FPS=8  # 降低帧率以减少处理量
BATCH_SIZE=1  # 降低批处理大小以减少内存使用

# 运行优化后的预处理
python -m data.precomp_video \
    --videos_dir "$INPUT_DIR" \
    --save_dir "$OUTPUT_DIR" \
    --vae_weight_path "$VAE_WEIGHTS" \
    --fps "$FPS" \
    --batch_size "$BATCH_SIZE"
EOF

chmod +x "${PRECOMP_SCRIPT}"

echo "优化配置已完成!"
echo "1. 使用优化后的配置文件: ${OPTIMIZED_CONFIG}"
echo "2. 使用优化后的训练脚本: ${OPTIMIZED_SCRIPT}"
echo "3. 使用优化后的预处理脚本: ${PRECOMP_SCRIPT}"
echo ""
echo "运行训练: bash ${OPTIMIZED_SCRIPT}"
echo "运行预处理: bash ${PRECOMP_SCRIPT} [输入视频目录] [输出目录] [VAE权重路径]"