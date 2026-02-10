import torch
import sys
import os
import subprocess

print("=" * 60)
print("GPU环境终极诊断")
print("=" * 60)

# 1. 系统信息
print("1. 系统信息:")
print(f"   Python版本: {sys.version}")
print(f"   操作系统: {sys.platform}")

# 2. PyTorch信息
print("\n2. PyTorch信息:")
print(f"   PyTorch版本: {torch.__version__}")
print(f"   PyTorch安装位置: {torch.__file__}")

# 3. CUDA信息
print("\n3. CUDA信息:")
print(f"   CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   设备 {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"     计算能力: {props.major}.{props.minor}")
        print(f"     显存: {props.total_memory / 1e9:.2f} GB")
else:
    print("   CUDA不可用")

# 4. 检查可能的PyTorch变体
print("\n4. PyTorch变体检查:")
try:
    import torch.backends.cudnn
    print(f"   cuDNN版本: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else 'N/A'}")
except:
    print("   无法获取cuDNN信息")

# 5. 环境变量检查
print("\n5. 环境变量检查:")
env_vars = ['CUDA_VISIBLE_DEVICES', 'TORCH_CUDA_ARCH_LIST', 'PATH']
for var in env_vars:
    value = os.environ.get(var, '未设置')
    print(f"   {var}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")

# 6. 运行外部nvidia-smi命令
print("\n6. NVIDIA驱动检查:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines[:8]):  # 只显示前8行
            print(f"   {line}")
        if len(lines) > 8:
            print("   ... (输出截断)")
    else:
        print(f"   nvidia-smi失败: {result.stderr}")
except FileNotFoundError:
    print("   nvidia-smi命令不存在，可能未安装NVIDIA驱动")

# 7. 尝试创建简单的CUDA张量
print("\n7. CUDA张量测试:")
if torch.cuda.is_available():
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y
        print(f"   CUDA张量测试: 成功 → {z}")
    except Exception as e:
        print(f"   CUDA张量测试: 失败 → {e}")
else:
    print("   CUDA不可用，跳过测试")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)