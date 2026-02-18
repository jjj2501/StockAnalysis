@echo off
echo ========================================
echo StockAnalysis AI 服务器启动
echo ========================================
echo.

REM 使用当前环境的Python
echo 检查Python环境...
python --version
if errorlevel 1 (
    echo [ERROR] Python未找到或无法运行
    pause
    exit /b 1
)

echo.
echo 初始化数据库...
python scripts\init_db.py
if errorlevel 1 (
    echo [WARNING] 数据库初始化失败，继续启动...
)

echo.
echo 启动服务器...
echo 访问地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo 前端界面: http://localhost:8000
echo.
echo 按 Ctrl+C 停止服务器
echo.

cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000

echo.
echo 服务器已停止
pause