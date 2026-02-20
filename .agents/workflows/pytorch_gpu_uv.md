---
description: [如何在 uv 管理的 Python 3.10 项目中正确安装并配置 PyTorch CUDA 版本 (如 2.5.1+cu121)]
---

# 在 uv 项目中配置 PyTorch CUDA 版本

在使用 `uv` 管理的项目中，直接使用 `uv add torch` 默认会拉取 PyTorch 的 CPU 版本，导致无法使用 GPU 加速。为了在 Python 3.10 环境下稳定运行特定版本的 GPU (CUDA) 版 PyTorch，请遵循以下流程。

## 适用场景
- 开发环境检测到了 NVIDIA 显卡（如 RTX 3050）
- 需要使用 CUDA 加速深度学习模型
- 前端设备自动检测显示 "CPU 模式" 而不是 "CUDA 模式"

## 配置步骤

### 1. 修改 `pyproject.toml`
这是最关键的一步，必须通过 `[tool.uv]` 配置显式指定 PyTorch 的 CUDA wheel 下载索引，否则 `uv` 依然会走 PyPI 的默认源。

在项目的 `pyproject.toml` 的末尾追加以下内容：

```toml
[tool.uv]
# 定义 PyTorch CUDA 12.1 源 (以 cu121 为例)
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

# 将这三个核心库的解析源指向定义好的 CUDA 源
[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
```

### 2. 指定确切的版本和 CUDA 后缀
同样在 `pyproject.toml` 的 `dependencies` 列表中，明确指定要求的版本。

以 PyTorch 2.5.1 为例（注意不需要在这里写 `+cu121`，因为 `[tool.uv.sources]` 会确保从此源获取）：

```toml
dependencies = [
    # ... 其他依赖 ...
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
]
```

### 3. 同步环境变量
执行 `uv sync` 命令，使 uv 解析依赖树并下载正确版本的 whl 文件。

```bash
uv sync
```
*注意：CUDA 版的 PyTorch 体积约 2.4GB，下载通常需要几分钟。*

### 4. 彻底重启相关服务
如果在 VSCode 终端、或后端服务器已经处于运行状态（如 `uv run python backend/main.py`），缓存旧版本的 Python 进程依然占用。
必须：
1. `Ctrl+C` 停止运行中的 Python 服务。
2. （如果遇到锁死端口）强制终止：`Stop-Process -Name "python" -Force`
3. 重新启动服务：`uv run python backend/main.py`

## 验证与测试
可在命令行中快速验证：
```powershell
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Version:', torch.__version__)"
```
如果输出包含 `CUDA available: True`，且版本以 `+cu121` 结尾，则配置成功！可以随时复用到其他使用了 `uv` 的 AI 项目中。
