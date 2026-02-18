# StockAnalysis AI - 快速启动指南

## 🚀 一键启动

### 方法1：使用Python启动脚本（推荐）
```bash
python run_server.py
```

### 方法2：直接运行服务器
```bash
python test_server.py
```

## 🌐 访问系统

### 主要页面
1. **登录页面**: http://localhost:8080/frontend/login.html
2. **主仪表盘**: http://localhost:8080/frontend/index.html  
3. **注册页面**: http://localhost:8080/frontend/register.html
4. **设置页面**: http://localhost:8080/frontend/settings.html
5. **自选股页面**: http://localhost:8080/frontend/watchlist.html

### 测试账户
- **用户名**: `testuser`
- **密码**: `password123`

## 📊 系统状态检查

### API接口
```bash
# 服务器健康状态
curl http://localhost:8080/api/health

# 可用股票列表
curl http://localhost:8080/api/stocks

# 单个股票信息（如AAPL）
curl http://localhost:8080/api/stock/AAPL
```

### 用户注册测试
```bash
curl -X POST http://localhost:8080/api/register \
  -H "Content-Type: application/json" \
  -d '{"username":"newuser","email":"new@example.com","password":"mypassword123","full_name":"New User"}'
```

## ⚙️ 服务器管理

### 查看运行状态
```bash
# 查看占用8080端口的进程
netstat -ano | findstr :8080

# 查看Python进程
tasklist | findstr python
```

### 停止服务器
1. 在运行服务器的命令行窗口按 **Ctrl+C**
2. 或使用任务管理器终止Python进程

### 重启服务器
```bash
# 停止当前服务器后
python run_server.py
```

## 🔧 故障排除

### 问题：端口8080被占用
**解决方案**：
```bash
# 使用启动脚本自动处理
python run_server.py

# 或手动查找并终止进程
netstat -ano | findstr :8080
taskkill /PID <进程ID> /F
```

### 问题：无法访问页面
**检查步骤**：
1. 确认服务器正在运行
2. 检查防火墙设置
3. 使用完整URL：http://localhost:8080/frontend/login.html

### 问题：数据库错误
**解决方案**：
1. 删除 `stockanalysis_test.db` 文件
2. 重启服务器（会自动重建数据库）

## 📁 文件说明

### 核心文件
- `test_server.py` - 服务器主程序
- `run_server.py` - 自动启动脚本
- `stockanalysis_test.db` - 数据库文件

### 前端文件
- `frontend/` - 所有界面文件
- `frontend/style.css` - 样式文件
- `frontend/app.js` - 前端逻辑

### 文档文件
- `README_FINAL.md` - 完整使用说明
- `DEPLOYMENT_COMPLETE.md` - 部署报告
- `QUICK_START.md` - 本快速指南

## 🎯 使用流程

### 第一步：启动服务器
```bash
python run_server.py
```

### 第二步：打开浏览器
访问：http://localhost:8080/frontend/login.html

### 第三步：登录或注册
- 使用测试账户：`testuser` / `password123`
- 或注册新账户

### 第四步：开始分析
1. 输入股票代码（如：AAPL、MSFT、GOOGL）
2. 点击"开始分析"
3. 查看AI分析报告和图表

## 📞 技术支持

### 查看服务器日志
服务器运行时会在控制台显示详细日志。

### 验证系统功能
```bash
# 验证API
curl http://localhost:8080/api/health

# 验证前端
curl -I http://localhost:8080/frontend/login.html
```

### 重置系统
1. 停止服务器
2. 删除 `stockanalysis_test.db`
3. 重新启动服务器

---

**启动命令**: `python run_server.py`

**访问地址**: http://localhost:8080/frontend/login.html

**默认账户**: 用户名 `testuser`，密码 `password123`

*系统已就绪，可以开始使用！*