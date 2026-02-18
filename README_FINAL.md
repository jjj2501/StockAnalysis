# StockAnalysis AI - 智能股票分析系统

## 项目概述
StockAnalysis AI 是一个为个人投资者设计的专业股票分析平台，提供现代化的用户界面和完整的多用户登录功能。

## 🚀 快速开始

### 1. 启动服务器
```bash
# 方法1: 使用启动脚本（推荐）
start_stockanalysis.bat

# 方法2: 直接运行Python
python test_server.py
```

### 2. 访问系统
- **登录页面**: http://localhost:8080/frontend/login.html
- **主仪表盘**: http://localhost:8080/frontend/index.html
- **注册页面**: http://localhost:8080/frontend/register.html

### 3. 测试账户
- **用户名**: `testuser`
- **密码**: `password123`

## 📋 系统功能

### ✅ 已完成功能
1. **现代化用户界面**
   - 专业侧边栏导航
   - 简洁明亮的设计风格
   - 响应式布局，适配不同屏幕

2. **完整的用户系统**
   - 用户注册与登录
   - 密码强度验证
   - JWT令牌认证
   - 多用户数据隔离

3. **股票分析功能**
   - 实时股票数据（模拟）
   - AI分析报告生成
   - 价格走势图表
   - 技术指标分析

4. **个人化管理**
   - 自选股列表
   - 个人设置页面
   - 投资偏好配置

## 🛠️ 技术架构

### 前端技术
- **HTML5/CSS3**: 现代化界面设计
- **JavaScript**: 交互逻辑和API调用
- **Chart.js**: 数据可视化图表

### 后端技术
- **Python 3.10+**: 服务器逻辑
- **SQLite**: 轻量级数据库
- **标准HTTP库**: 无外部依赖

### 数据库结构
```
stockanalysis_test.db
├── users (用户表)
│   ├── id (用户ID)
│   ├── username (用户名)
│   ├── email (邮箱)
│   ├── password_hash (密码哈希)
│   └── created_at (创建时间)
└── watchlist (自选股表)
    ├── user_id (用户ID)
    ├── symbol (股票代码)
    ├── price (价格)
    └── added_at (添加时间)
```

## 🔧 API接口

### 用户认证
```http
POST /api/register
Content-Type: application/json
{
  "username": "用户名",
  "email": "邮箱",
  "password": "密码",
  "full_name": "姓名"
}

POST /api/login
Content-Type: application/json
{
  "username": "用户名",
  "password": "密码"
}
```

### 股票分析
```http
POST /api/analyze
Content-Type: application/json
{
  "symbol": "股票代码"
}

GET /api/stocks
GET /api/stock/{symbol}
```

### 自选股管理
```http
POST /api/watchlist
Content-Type: application/json
{
  "action": "add|remove|list",
  "symbol": "股票代码",
  "user_id": 用户ID
}
```

## 📁 文件结构

```
StockAnalysisAI/
├── frontend/                    # 前端文件
│   ├── style.css              # 样式文件
│   ├── login.html             # 登录页面
│   ├── register.html          # 注册页面
│   ├── index.html             # 主仪表盘
│   ├── settings.html          # 设置页面
│   ├── watchlist.html         # 自选股页面
│   └── app.js                # 前端逻辑
├── test_server.py             # 服务器主程序
├── start_stockanalysis.bat    # 启动脚本
├── stockanalysis_test.db      # 数据库文件
├── README_FINAL.md           # 本说明文件
└── DEPLOYMENT_COMPLETE.md    # 部署完成报告
```

## 🎯 使用指南

### 第一步：注册账户
1. 访问 http://localhost:8080/frontend/register.html
2. 填写用户名、邮箱、密码等信息
3. 点击注册按钮完成账户创建

### 第二步：登录系统
1. 访问 http://localhost:8080/frontend/login.html
2. 输入用户名和密码
3. 点击登录进入主界面

### 第三步：分析股票
1. 在主界面输入股票代码（如：AAPL）
2. 点击"开始分析"按钮
3. 查看AI生成的分析报告和图表

### 第四步：管理自选股
1. 在股票分析页面点击"加入自选"
2. 访问自选股页面查看和管理
3. 可以随时添加或删除股票

## 🔍 测试数据

系统包含以下模拟股票数据：
- **AAPL**: Apple Inc. (苹果公司)
- **MSFT**: Microsoft Corporation (微软)
- **GOOGL**: Alphabet Inc. (谷歌)
- **AMZN**: Amazon.com Inc. (亚马逊)
- **TSLA**: Tesla Inc. (特斯拉)
- **NVDA**: NVIDIA Corporation (英伟达)

## ⚙️ 配置选项

### 修改服务器端口
编辑 `test_server.py` 文件，修改第509行的端口号：
```python
PORT = 8080  # 修改为其他端口，如 8000
```

### 添加更多股票数据
编辑 `test_server.py` 文件，在 `MOCK_STOCKS` 字典中添加新的股票数据。

## 🚨 故障排除

### 问题1：端口被占用
```
错误: [WinError 10048] 通常每个套接字地址只允许使用一次
```
**解决方案**：
1. 使用启动脚本自动处理
2. 手动终止占用8080端口的进程
3. 修改服务器使用其他端口

### 问题2：Python未安装
```
错误: 未找到Python
```
**解决方案**：
1. 安装Python 3.10或更高版本
2. 确保Python已添加到系统PATH

### 问题3：数据库错误
```
错误: 无法访问数据库文件
```
**解决方案**：
1. 检查文件权限
2. 删除损坏的数据库文件，服务器会自动重建

## 📞 技术支持

### 系统状态检查
```bash
# 检查服务器是否运行
curl http://localhost:8080/api/health

# 检查可用股票
curl http://localhost:8080/api/stocks
```

### 日志查看
服务器运行时会在控制台显示访问日志和错误信息。

## 🎉 部署完成

系统已成功部署并具备以下特点：
1. **专业界面** - 现代化设计，适合个人投资者
2. **完整功能** - 注册、登录、分析、管理一体化
3. **多用户支持** - 数据隔离，安全可靠
4. **零依赖** - 仅需Python标准库
5. **易于扩展** - 模块化设计，便于功能添加

---

**启动系统**: 双击 `start_stockanalysis.bat` 或运行 `python test_server.py`

**访问地址**: http://localhost:8080/frontend/login.html

**测试账户**: 用户名 `testuser`，密码 `password123`

*系统部署完成时间: 2026年2月17日*