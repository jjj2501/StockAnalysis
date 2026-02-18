# StockAnalysis AI - 智能选股工具

基于 Transformer + LSTM 混合模型与 LLM (Ollama) 的 A 股智能分析工具。

## 功能特性

- **多模态分析**: 结合数值预测 (Hybrid Model) 与 语义分析 (LLM)。
- **混合神经网络**: Transformer (全局趋势) + LSTM (局部时序)。
- **本地 LLM 集成**: 支持 Ollama (默认 qwen2.5) 生成中文投资日报。
- **现代化界面**: 极简深色模式 UI，交互流畅。

## 安装与运行

### 前置要求

1. Python 3.10+
2. [Ollama](https://ollama.com/) (建议模型: `qwen2.5:7b-instruct`)
   - 运行 `ollama pull qwen2.5:7b-instruct` 下载模型
   - 运行 `ollama serve` 启动服务

### 1. 初始化环境

如果尚未安装 uv:
```bash
pip install uv
```

安装依赖:
```bash
uv sync
```
(或者如果不使用 uv，请查看 pyproject.toml 安装对应库)

### 2. 启动服务

```bash
uv run python -m backend.main
```
或者直接使用虚拟环境中的 python:
```bash
.venv\Scripts\python -m backend.main
```

### 3. 使用

访问浏览器: [http://localhost:8000](http://localhost:8000)

1. 打开网页，输入 A 股代码 (如 `600519`)。
2. 点击"开始分析"。
   - 首次对某只股票分析时，系统会自动下载数据并在后台训练模型 (可能需要几秒钟)。
3. 查看预测结果、走势图及 AI 生成的分析日报。

## 用户认证系统

系统现已集成完整的用户认证和管理功能：

### 认证功能
- **用户注册/登录**: JWT令牌认证
- **密码管理**: 密码重置、修改密码
- **会话管理**: 访问令牌和刷新令牌
- **权限控制**: 角色基础权限系统（免费用户、高级用户、管理员）
- **安全特性**: Argon2密码哈希、速率限制、审计日志

### API端点
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录
- `POST /api/auth/refresh` - 刷新访问令牌
- `POST /api/auth/logout` - 用户登出
- `GET /api/users/me` - 获取当前用户信息
- `POST /api/auth/forgot-password` - 忘记密码
- `POST /api/auth/reset-password` - 重置密码

### 数据库初始化
首次运行前需要初始化数据库：
```bash
python scripts/init_db.py
```

### 测试认证系统
```bash
python scripts/test_complete_auth.py
```

## GPU训练支持

系统现已全面支持GPU加速训练，提供显著的性能提升：

### GPU特性
- **自动GPU检测**: 自动检测CUDA可用性并选择最佳设备
- **智能内存管理**: 动态内存分配和缓存清理
- **批处理优化**: 自动调整批次大小避免内存溢出
- **混合精度训练**: 支持FP16混合精度训练加速
- **多GPU支持**: 支持数据并行多GPU训练
- **完整API管理**: 通过REST API管理GPU和训练设置

### GPU配置
```bash
# 启用GPU支持
export USE_GPU=True
export GPU_DEVICE_ID=0
export BATCH_SIZE=32

# 训练参数
export EPOCHS=50
export LEARNING_RATE=0.001
export MODEL_HIDDEN_DIM=64
```

### GPU API端点
- `GET /api/gpu/status` - 获取GPU状态
- `GET /api/gpu/devices` - 获取GPU设备列表
- `GET /api/gpu/memory` - 获取GPU内存信息
- `POST /api/gpu/config/update` - 更新训练配置
- `POST /api/gpu/train` - 启动GPU训练
- `GET /api/gpu/training/status` - 获取训练状态
- `POST /api/gpu/clear-cache` - 清除GPU缓存
- `GET /api/gpu/benchmark` - 运行性能基准测试

### 测试GPU功能
```bash
# 运行GPU功能测试
python scripts/test_gpu_training.py

# 检查GPU状态
python -c "from backend.core.gpu_utils import device_info; import json; print(json.dumps(device_info(), indent=2))"
```

### 性能提升
- **训练加速**: 6-7倍性能提升（GPU vs CPU）
- **推理加速**: 5-7倍性能提升
- **内存优化**: 智能批次大小调整避免OOM错误

## 目录结构

- `backend/`: 后端核心代码 (FastAPI, PyTorch)
  - `core/`: 数据、模型、引擎、LLM
    - `engine.py`: 训练引擎（支持GPU加速）
    - `model.py`: HybridModel（Transformers + LSTM）
    - `data.py`: 数据获取和预处理
    - `gpu_utils.py`: GPU管理工具（新增）
  - `api/`: 接口路由
    - `enhanced_router.py`: 增强API路由
    - `data_router.py`: 数据API路由
    - `gpu_router.py`: GPU管理API（新增）
  - `auth/`: 认证系统
    - `models.py`: 数据库模型
    - `schemas.py`: Pydantic验证模式
    - `security.py`: 安全工具
    - `routers/`: 认证和用户路由
    - `middleware.py`: 认证中间件
  - `config.py`: 应用配置（已更新支持GPU）
- `frontend/`: 前端静态资源
  - `index.html`: 主页面（已更新）
  - `login.html`: 登录页面
  - `register.html`: 注册页面
  - `dashboard.html`: 用户仪表板
  - `style.css`: 样式表（已更新）
  - `app.js`: JavaScript（已更新）
- `backend/models/`: 训练好的模型权重保存目录
- `scripts/`: 工具脚本
  - `init_db.py`: 数据库初始化
  - `test_auth.py`: 认证系统测试
  - `test_complete_auth.py`: 完整认证流程测试
  - `test_gpu_training.py`: GPU训练功能测试（新增）
  - `start_server.py`: 服务器启动脚本
- `docs/`: 文档
  - `AUTHENTICATION.md`: 认证系统文档
  - `GPU_SUPPORT.md`: GPU支持文档（新增）
- `DEPLOYMENT_SUMMARY.md`: 部署总结报告
