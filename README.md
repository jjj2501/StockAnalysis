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

## 目录结构

- `backend/`: 后端核心代码 (FastAPI, PyTorch)
  - `core/`: 数据、模型、引擎、LLM
  - `api/`: 接口路由
- `frontend/`: 前端静态资源
- `backend/models/`: 训练好的模型权重保存目录
