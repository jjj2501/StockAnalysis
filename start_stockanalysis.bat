@echo off
echo ========================================
echo StockAnalysis AI 系统启动脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.10+
    pause
    exit /b 1
)

REM 检查端口是否被占用
netstat -ano | findstr :8080 >nul
if not errorlevel 1 (
    echo 端口8080已被占用，正在查找占用进程...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do (
        echo 找到占用进程PID: %%a
        tasklist /FI "PID eq %%a" 2>nul
        echo.
        choice /c YN /m "是否终止进程并启动新服务器？(Y/N)"
        if errorlevel 2 (
            echo 取消启动
            pause
            exit /b 0
        ) else (
            echo 正在终止进程PID: %%a
            taskkill /PID %%a /F >nul 2>&1
            timeout /t 2 /nobreak >nul
        )
    )
)

echo 正在启动StockAnalysis AI服务器...
echo.
echo 访问地址: http://localhost:8080/frontend/login.html
echo 测试用户: testuser / password123
echo.
echo 按Ctrl+C停止服务器
echo ========================================
echo.

REM 启动服务器
python test_server.py

if errorlevel 1 (
    echo.
    echo 服务器启动失败，请检查错误信息
    pause
)