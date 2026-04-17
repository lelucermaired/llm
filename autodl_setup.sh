#!/bin/bash
# AutoDL环境配置脚本
# 用法: 在AutoDL实例上跑一次: bash autodl_setup.sh

set -e  # 出错立即停止

echo "================================================"
echo "AutoDL 环境配置 - LLM LoRA 研究项目"
echo "================================================"

# ========== 1. 配置HuggingFace国内镜像 ==========
echo ""
echo "[1/5] 配置HuggingFace镜像..."
cat >> ~/.bashrc << 'EOF'

# HuggingFace国内镜像 (AutoDL)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
EOF

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache

mkdir -p /root/autodl-tmp/hf_cache
echo "  ✓ HF镜像配置完成,缓存目录: /root/autodl-tmp/hf_cache"

# ========== 2. 配置pip国内源 ==========
echo ""
echo "[2/5] 配置pip国内源..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
EOF
echo "  ✓ pip源已切换到清华"

# ========== 3. 安装依赖 ==========
echo ""
echo "[3/5] 安装Python依赖..."
pip install --upgrade pip
pip install transformers peft trl datasets accelerate bitsandbytes safetensors
pip install matplotlib seaborn scikit-learn pandas  # 分析脚本用
echo "  ✓ 依赖安装完成"

# ========== 4. 下载基础模型 ==========
echo ""
echo "[4/5] 预下载Qwen2.5-7B-Instruct基础模型..."
echo "      (首次下载约15GB,视网速需要几分钟到半小时)"

python << 'PYEOF'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/hf_cache'

from huggingface_hub import snapshot_download
print("开始下载 Qwen/Qwen2.5-7B-Instruct ...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    cache_dir="/root/autodl-tmp/hf_cache",
)
print("✓ Qwen2.5-7B-Instruct 下载完成")
PYEOF

# ========== 5. GPU检查 ==========
echo ""
echo "[5/5] GPU环境检查..."
python << 'PYEOF'
import torch
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYEOF

echo ""
echo "================================================"
echo "✓ 配置完成! 建议执行: source ~/.bashrc"
echo "================================================"
echo ""
echo "下一步建议:"
echo "  1. cd /root/autodl-tmp/llm-project"
echo "  2. tmux new -s exp"
echo "  3. python diagnose_training_quality.py"
echo ""
