# GPU 环境配置说明

## 本机环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| 显存 | 4 GB |
| 驱动版本 | 591.74 |
| Python 版本 | 3.10 |
| 包管理器 | uv |

## PyTorch CUDA 安装

> [!IMPORTANT]
> 默认 `uv add torch` 会安装 **CPU 版** (`+cpu`)，不支持 GPU 加速。必须指定 CUDA wheel 源：

```bash
# 驱动 ≥ 527 支持 CUDA 12.x。使用 CUDA 12.4 wheel：
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 驱动 / CUDA 版本对应关系

| NVIDIA 驱动 | 支持的最高 CUDA | 推荐 PyTorch wheel |
|------------|---------------|------------------|
| ≥ 527.41 | CUDA 12.x | `cu124` |
| ≥ 452.39 | CUDA 11.x | `cu118` |

本机驱动 **591.74 → 最高支持 CUDA 12.x → 使用 `cu124`** ✅

## 验证 GPU 可用性

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 3050 Laptop GPU
print(torch.version.cuda)              # 12.4
```

命令行快速验证：

```powershell
.\uv.exe run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## uv 项目注意事项

- 用 `uv pip install`（而非 `uv add`）安装带自定义 index 的包
- `--index-url` 参数只对 `uv pip install` 有效
- 安装完成后重启后端服务使配置生效

## 项目 `.env` 配置

```env
USE_GPU=True
GPU_DEVICE_ID=0
BATCH_SIZE=64
```

> RTX 3050 4GB 显存建议 Batch Size 64，序列长度 60 以内。
