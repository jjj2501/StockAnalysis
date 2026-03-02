---
name: AlphaPulse 常见错误与解决方案速查手册 (Common Errors Skill)
description: 汇总了 AlphaPulse (StockAnalysis) 开发与运行过程中最常遇到的环境、网络、前后端及大模型对接错误及其标准解决方案。
---

# AlphaPulse 项目常见错误与解决方案速查手册

本技能文档汇总了 AlphaPulse 项目中开发者与用户常遇见的各类典型报错。在遇到类似报错时，请遵循此手册进行快速排查与修正逻辑。

## 1. 大模型 (LLM) 接入类错误

### 1.1 外部大脑短路: Model Not Exist (HTTP 400)
- **报错表现**: `[外部大脑短路: Error code: 400 - {'error': {'message': 'Model Not Exist', 'type': 'invalid_request_error'...}}]`
- **触发场景**: 在前端沙盘中使用了非本地（如 OpenAI / DeepSeek）的 `provider`，但此时前端向后端发起 SSE 请求时，未能在 URL 中附带当前模型名称（`modelName`参数），导致后端 `AgentOrchestrator` 错误使用了默认绑定给 Ollama 的 `qwen3:1.7b`，发往外部网关自然找不到该模型资源。
- **解决方案**: 
  1. 确保在系统设置页中选用了正确的外部模型名（如 `deepseek-chat` 或 `gpt-4o`）并保存。
  2. 检查 `web/src/routes/agents/+page.svelte` 的 EventSource 请求 URL 中是否成功拼接了 `&model=${modelName}` 参数。
  3. 检查后端路由 `backend/api/enhanced_router.py` 是否正常接收 `model` 参数并传入实例化函数。

### 1.2 外部大脑短路: Missing Field `id` (工具调用序列化失败)
- **报错表现**: `[外部大脑短路: Error code: 400 - {'error': {'message': 'Failed to deserialize the JSON body into the target type: messages[3]: missing field `id`...}}]`
- **触发场景**: 采用了严格遵守 OpenAI Tool Calling / MCP 协议的外部大语言模型。当大模型下发拉取数据的 Function Call 请求时，附带了随机的序列化辨识码 `tool_call_id`。如果后端的 ReAct 沙盘沙盒在执行完探针代码向大模型回复结果时，构建的 `{"role": "tool", "content": "..."}` JSON 结构中漏传了这个标识，则会引发强校验体系的崩溃。
- **解决方案**:
  1. 检查底层网关 `backend/core/llm.py` 内的 `message_obj.tool_calls` 解析逻辑，确保原封不动提取了 `id` 与 `type='function'` 参数。
  2. 检查调度中心 `backend/core/agents/base.py` 中拼接 `role: tool` 消息记录的回传语句，必须补充挂载提取到的 `tool_call_id`。

## 2. 前后端数据通信错误

### 2.1 用户注册/登录 405 Method Not Allowed
- **报错表现**: 前台点击登录或注册抛出 405 错误，前端控制台出现 `fetch` Failed 打点异常。
- **触发场景**: 前端硬编码请求的是旧版本 API 端点（例如 `/api/register`），但实际的 FastAPI 路由系统已经划分并挂载了专属前缀的路由组群（例如使用了 `prefix="/api/auth"`，故实际路径应为 `/api/auth/register`）。
- **解决方案**:
  1. 检视 `backend/main.py` 入口文件或 `backend/auth/router.py` 中的 `prefix` 设定。
  2. 统一排查前端的所有 HTML/JS 或 Svelte 代码中遭遇 405 的地址并将其修正对接至正确前缀。

### 2.2 股票指标查询引发的数据格式报错 
- **报错表现**: Python 引擎抛出 `ValueError: could not convert string to float: 'AAPL'`。
- **触发场景**: 在量化工程的预测训练或因子搜集过程中，由于数据拉取工具（如 `yfinance`）返回的 Pandas DataFrame 的索引结构或多列命名行为发生改变，导致下层程序将字符串当作有效特征值送往了张量数学中心。
- **解决方案**: 在 `DataFetcher` 或者是模型预处理模块（如 `clean_data`）部分，使用 `.select_dtypes(include=[np.number])` 强制筛除含有字符串因子的不合法列，以及在清洗阶段用 `pd.to_numeric` 的错误抑制函数进行托底。

## 3. 网络与代理类错误 (ProxyError)

### 3.1 爬虫模块遭代理阻断 (ProxyError)
- **报错表现**: 触发 `requests.exceptions.ProxyError` ，导致 `yfinance` 或 AKShare 行情接口卡死并返回失败。
- **触发场景**: 这是中国大陆局域网环境中最常面临的痛点。主要源于开发机器本地或者系统注入了网络代理，接管了所有的 HTTP/HTTPS 进程。原生的 Python 底层库如 urllib 或 socket 在解析默认信道时因为代理端口闭锁或失效而无法穿梭握手。
- **解决方案**: 
  1. 在全局发车入口或是发起网络请求函数的入口处（如 `backend/core/data.py`），最简单粗暴地加入环境变量短路隔离机制让 Python 直连物理网卡：
     ```python
     import os
     os.environ['NO_PROXY'] = '*'
     ```
  2. 也可以在特定方法调用时通过局部显式地给 `requests` 传空字典强制走直连：`proxies={"http": None, "https": None}`。

## 4. 环境部署与运行错误

### 4.1 Windows PowerShell 下无法识别 uv 命令
- **报错表现**:
  ```powershell
  uv : 无法将“uv”项识别为 cmdlet、函数、脚本文件或可运行程序的名称。
  Suggestion [3,General]: 找不到命令 uv，但它确实存在于当前位置。默认情况下，Windows PowerShell 不会从当前位置加载命令。如果信任此命令，请改为键入“.\uv”。
  ```
- **触发场景**: 在 Windows PowerShell 终端中，尝试直接执行当前目录下的 `uv` (例如 `uv run uvicorn`)，但由于 Windows 默认的安全策略限制，不会自动执行当前目录的可执行文件，导致抛出命令未找到的异常。
- **解决方案**:
  1. **明确路径执行**: 按提示在命令前加上 `.\`，例如 `.\uv run uvicorn ...`
  2. **直接使用虚拟环境的 Python**: 绕过 uv 显式调用虚拟环境内的可执行程序，如 `.venv\Scripts\python.exe -m uvicorn ...` 或直接 `.venv\Scripts\pip.exe install ...`

## 使用指引
本速查手册技能项将随项目推进动态演进。当系统抛出不明断言阻断流或者引发界面不可知崩溃时，请首先比对本技能册尝试采用标准防务方案进行恢复。
