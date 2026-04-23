#!/bin/bash
# PyTorch环境激活脚本 (GPU加速版 - v2.11.0)
# 使用方法: source setup_env.sh

echo "======================================"
echo "激活PyTorch虚拟环境 (GPU加速)"
echo "版本: PyTorch 2.11.0 + CUDA 13.0 ⭐"
echo "======================================"

# 激活虚拟环境
source venv/bin/activate

echo "✓ 虚拟环境已激活"
echo ""
echo "📊 环境信息:"
echo "  Python版本: $(python --version)"
echo "  PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# 检查GPU支持
python -c "
import torch
if torch.cuda.is_available():
    print('🎮 GPU支持:')
    print(f'  ✓ CUDA {torch.version.cuda} 可用')
    print(f'  ✓ 设备: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'  ✓ GPU数量: {torch.cuda.device_count()}')
elif torch.backends.mps.is_available():
    print('🍎 MPS可用 (Apple Silicon)')
else:
    print('⚠️  使用CPU (未检测到GPU)')
"

echo ""
echo "--------------------------------------"
echo "快速开始:"
echo "  简单Demo: python simple_demo.py"
echo "  完整Demo: python pytorch_demo.py"
echo "  性能测试: python gpu_benchmark.py"
echo "  GPU指南:  cat GPU_GUIDE.md"
echo "  更新日志: cat CHANGELOG.md"
echo "--------------------------------------"
echo ""
