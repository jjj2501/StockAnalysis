@echo off
echo ========================================
echo StockAnalysis AI 系统部署测试
echo ========================================
echo.

REM 设置Python路径
set PYTHON_PATH=C:\Users\billb\AppData\Local\Python\bin\python.exe

echo 步骤1: 检查Python环境
%PYTHON_PATH% --version
if errorlevel 1 (
    echo [ERROR] Python未找到或无法运行
    pause
    exit /b 1
)

echo.
echo 步骤2: 初始化数据库
%PYTHON_PATH% scripts\init_db.py
if errorlevel 1 (
    echo [ERROR] 数据库初始化失败
    pause
    exit /b 1
)

echo.
echo 步骤3: 启动服务器（后台运行）
start "StockAnalysis Server" %PYTHON_PATH% -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
timeout /t 5 /nobreak > nul

echo.
echo 步骤4: 测试服务器连接
curl -s -o nul -w "%%{http_code}" http://localhost:8000/docs
if errorlevel 1 (
    echo [ERROR] 服务器未响应
    pause
    exit /b 1
)

echo.
echo 步骤5: 运行API测试
%PYTHON_PATH% scripts\test_api_live.py
if errorlevel 1 (
    echo [WARNING] API测试失败，继续前端测试
)

echo.
echo 步骤6: 运行前端测试
%PYTHON_PATH% scripts\test_frontend_flow.py

echo.
echo ========================================
echo 部署测试完成！
echo ========================================
echo.
echo 访问地址:
echo  前端: http://localhost:8000
echo  API文档: http://localhost:8000/docs
echo.
echo 按任意键停止服务器并退出...
pause > nul

REM 停止服务器
taskkill /F /IM python.exe /T > nul 2>&1
taskkill /F /IM uvicorn.exe /T > nul 2>&1

echo 服务器已停止
pause