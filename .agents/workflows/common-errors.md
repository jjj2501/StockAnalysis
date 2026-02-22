---
description: AlphaPulse 项目常见错误与解决方案速查手册
---
// turbo-all

# AlphaPulse 项目常见错误与解决方案

本文档汇总了项目开发过程中频繁遇到的错误，按类别分组，每个错误包含**报错信息**、**根因分析**和**解决方案**。

---

## 一、Windows PowerShell 环境问题

### 1.1 `&&` 操作符不支持

**报错信息：**
```
标记"&&"不是此版本中的有效语句分隔符。
+ CategoryInfo: ParserError: (:) [], ParentContainsErrorRecordException
+ FullyQualifiedErrorId: InvalidEndOfLine
```

**根因：** Windows PowerShell (5.x) 不支持 `&&` 链式命令操作符，这是 Bash / PowerShell 7+ 的语法。

**解决方案：** 使用分号 `;` 代替 `&&`，或者分两条命令执行。
```powershell
# ❌ 错误
cd c:\project && npm run dev

# ✅ 正确
cd c:\project; npm run dev
```

> **给 Agent 的提示：** 在 run_command 工具中，避免使用 `&&`。应将命令拆分为独立调用，或使用 `;` 连接。

---

### 1.2 `uv` 命令找不到

**报错信息：**
```
Suggestion [3,General]: 找不到命令 uv，但它确实存在于当前位置。
默认情况下，Windows PowerShell 不会从当前位置加载命令。
如果信任此命令，请改为键入".\uv"。
```

**根因：** Windows PowerShell 的安全策略不允许直接运行当前目录下的可执行文件，必须显式指定路径。

**解决方案：**
```powershell
# ❌ 报错
uv run python script.py

# ✅ 方案 1：用 .\uv
.\uv run python script.py

# ✅ 方案 2：用完整路径（推荐，更可靠）
# 确保 uv 已添加到系统 PATH 环境变量中
# 通常 uv 安装后位于: %USERPROFILE%\.cargo\bin\uv.exe
```

> **给 Agent 的提示：** 本项目要求使用 `uv` 管理 Python 依赖。如果 `uv` 在 PATH 中可用，直接使用 `uv run python ...`；否则使用 `.\uv` 或完整路径。安装库用 `uv add <package>` 或 `uv pip install <package>`。

---

### 1.3 Python 命令行执行超时/挂起

**现象：** `uv run python -c "..."` 或 `python -c "..."` 执行后长时间无输出。

**可能原因：**
1. **uv 首次启动缓慢** — uv 需要解析虚拟环境和依赖关系
2. **网络请求阻塞** — 脚本中包含网络请求（如 akshare、yfinance）且网络不通或被代理拦截
3. **模块导入缓慢** — 某些大型库（如 torch、pandas）首次导入耗时较长

**解决方案：**
- 设置合理的超时时间（WaitMsBeforeAsync 设为 10000+）
- 对网络请求添加超时控制（见 2.1）
- 使用 `send_command_input` 的 `Terminate` 终止挂起的命令

---

## 二、后端 Python / FastAPI 问题

### 2.1 AkShare 数据获取超时

**现象：** `ak.stock_zh_a_hist()` 调用后长时间无响应，导致 API 接口卡死返回超时。

**根因：** 系统代理设置干扰了 akshare 的网络请求，或国内数据源网络不稳定。

**解决方案：**
1. 在请求前清除代理环境变量（`_clear_proxy()` 方法）
2. 使用 `ThreadPoolExecutor` 实现超时控制：
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(ak.stock_zh_a_hist, symbol=symbol, ...)
    try:
        df = future.result(timeout=15)  # 15 秒超时
    except Exception:
        return pd.DataFrame()  # 超时后 fallback
```

---

### 2.2 缓存文件命名不匹配

**现象：** 本地已有数据（如 `600519.parquet`），但接口仍报 "No data found"。

**根因：** 缓存文件命名格式经历过升级（旧格式 `{symbol}.parquet` → 新格式 `{market}_{symbol}.parquet`），新代码无法识别旧缓存。

**解决方案：** 在 `_load_cache` 中添加 fallback 逻辑：
```python
# 优先查找新格式，其次查找旧格式
for path in [new_cache_path, legacy_cache_path]:
    if path.exists():
        return pd.read_parquet(path)
```

---

### 2.3 远程数据获取失败后缺少 fallback

**现象：** 缓存数据部分覆盖请求日期范围，尝试增量更新 → 网络失败 → 返回空 DataFrame → "No data found"。

**根因：** `get_stock_data` 增量更新分支在远程获取失败时没有 fallback 逻辑。

**解决方案：** 在每个远程获取失败的分支都添加 fallback 返回已有缓存：
```python
if not new_df.empty:
    # 合并新旧数据
    ...
else:
    # 远程失败时，返回已有缓存（数据虽不完整但比空结果好）
    return cached_df[(cached_df['date'] >= start) & (cached_df['date'] <= end)]
```

---

## 三、前端 SvelteKit 问题

### 3.1 Svelte 5 `$props()` 可选属性缺少默认值

**报错信息：**
```
Property 'header' is missing in type '{ children: () => any; }' but required in type '$$ComponentProps'.
```

**根因：** Svelte 5 的 `$props()` 解构中，没有默认值的属性会被 TypeScript 推断为必需的。

**解决方案：** 给可选属性提供默认值：
```svelte
<!-- ❌ header 被视为必需 -->
let { children, title = "", header, class: className = "" } = $props();

<!-- ✅ header 变为可选 -->
let { children, title = "", header = undefined, class: className = "" } = $props();
```

---

### 3.2 SvelteKit 嵌套布局叠加问题

**现象：** 登录页面（`/auth/login`）本应独立全屏显示，但仍然显示侧边栏和顶栏。

**根因：** SvelteKit 的嵌套布局是**累加性的**：子布局 `auth/+layout.svelte` 在父布局 `+layout.svelte` 内部渲染，因此侧边栏始终可见。

**解决方案：** 在根布局中根据路由判断布局类型：
```svelte
<script>
    let isAuthPage = $derived($page.url.pathname.startsWith("/auth"));
</script>

{#if isAuthPage}
    <div class="min-h-screen">
        {@render children()}
    </div>
{:else}
    <div class="flex min-h-screen">
        <Sidebar />
        <main>{@render children()}</main>
    </div>
{/if}
```

---

### 3.3 硬编码 API URL 导致 CORS 问题

**报错信息：** 浏览器控制台出现 CORS 跨域错误，如：
```
Access to fetch at 'http://localhost:8000/api/...' from origin 'http://127.0.0.1:3000' has been blocked by CORS policy
```

**根因：** 前端使用绝对 URL `http://localhost:8000/api/...` 调用后端，跨端口视为跨域。

**解决方案：**
1. **前端统一使用相对路径**：
```javascript
// ❌ 硬编码
fetch('http://localhost:8000/api/backtest/600519')

// ✅ 相对路径
fetch('/api/backtest/600519')
```

2. **Vite 配置代理**（`vite.config.js`）：
```javascript
server: {
    proxy: {
        '/api': {
            target: 'http://localhost:8000',
            changeOrigin: true,
        }
    }
}
```

---

### 3.4 TypeScript 类型错误：`$state` 声明

**报错信息：**
```
Type 'HTMLCanvasElement | null' is not assignable to type 'HTMLCanvasElement'.
```

**根因：** Svelte 5 的 DOM 引用绑定（`bind:this`）在初始化时为 `null`，TypeScript 严格模式下类型不匹配。

**解决方案：** 使用 JSDoc 类型注解声明：
```svelte
<script>
    /** @type {HTMLCanvasElement} */
    let chartCanvas = $state();
    
    /** @type {import('chart.js').Chart | null} */
    let chartInstance = $state(null);
</script>
```

---

## 四、API 路由匹配问题

### 4.1 405 Method Not Allowed

**现象：** 前端调用注册/登录接口返回 405。

**根因：** API 路径不匹配。后端路由使用了 prefix 组合：
- `app.include_router(auth_router, prefix="/api")`
- `router = APIRouter(prefix="/auth")`
- 最终路径：`/api/auth/login`，而非 `/api/login`

**解决方案：** 检查后端 `main.py` 中的 router prefix 和各 router 自身的 prefix，确保前端调用路径完全匹配。

**快速验证方法：**
```python
# 列出所有注册的路由
for route in app.routes:
    if hasattr(route, 'path'):
        print(f"{route.methods} {route.path}")
```

---

## 五、Git / 版本控制问题

### 5.1 PowerShell 中 Git 命令编码问题

**现象：** `git log` 输出中文出现乱码。

**解决方案：**
```powershell
$env:LANG="zh_CN.UTF-8"
git -c core.quotepath=false log --oneline -10
```

---

## 总结：Agent 操作最佳实践

| 场景 | 推荐做法 |
|---|---|
| 执行 Shell 命令 | 使用 `;` 连接命令，不用 `&&` |
| 运行 Python | `uv run python script.py`，确保 uv 在 PATH |
| 网络数据获取 | 添加超时控制 + 本地缓存 fallback |
| Svelte 组件 props | 可选属性必须提供默认值 |
| API 调用 | 前端使用相对路径 + Vite proxy |
| 布局隔离 | 根布局根据路由条件渲染不同布局 |
