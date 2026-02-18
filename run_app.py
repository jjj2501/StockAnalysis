#!/usr/bin/env python3
"""
StockAnalysis AI 应用程序启动脚本
简化启动过程，处理依赖问题
"""

import sys
import os
import subprocess
import time
import webbrowser

def check_dependencies():
    """检查必要依赖"""
    print("检查依赖...")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("torch", "PyTorch"),
        ("pandas", "Pandas"),
        ("akshare", "AkShare"),
        ("sqlalchemy", "SQLAlchemy"),
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (缺失)")
            missing.append(module)
    
    return missing

def install_dependencies(missing_deps):
    """安装缺失的依赖"""
    if not missing_deps:
        return True
    
    print(f"\n安装缺失的依赖: {', '.join(missing_deps)}")
    
    try:
        # 安装核心依赖
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + missing_deps, check=True)
        
        print("依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")
        return False

def initialize_database():
    """初始化数据库"""
    print("\n初始化数据库...")
    
    try:
        from scripts.init_db import main as init_db_main
        init_db_main()
        return True
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        print("尝试手动初始化: python scripts/init_db.py")
        return False

def start_server():
    """启动服务器"""
    print("\n启动StockAnalysis AI服务器...")
    
    # 构建启动命令
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    print(f"启动命令: {' '.join(cmd)}")
    
    try:
        # 启动服务器进程
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 等待服务器启动
        print("等待服务器启动...")
        time.sleep(5)
        
        # 检查服务器是否运行
        import requests
        try:
            response = requests.get("http://localhost:8000/docs", timeout=5)
            if response.status_code == 200:
                print("✅ 服务器启动成功!")
                return server_process
            else:
                print(f"⚠️ 服务器返回状态码: {response.status_code}")
        except:
            print("⚠️ 服务器可能未完全启动，请检查输出")
        
        return server_process
        
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        return None

def open_browser():
    """打开浏览器"""
    print("\n在浏览器中打开应用程序...")
    
    urls = [
        ("前端界面", "http://localhost:8000"),
        ("API文档", "http://localhost:8000/docs"),
        ("登录页面", "http://localhost:8000/login.html"),
        ("GPU状态", "http://localhost:8000/api/gpu/status"),
    ]
    
    print("\n访问地址:")
    for name, url in urls:
        print(f"  {name}: {url}")
    
    try:
        # 打开前端界面
        webbrowser.open("http://localhost:8000")
        time.sleep(1)
        
        # 打开API文档
        webbrowser.open("http://localhost:8000/docs")
        
    except Exception as e:
        print(f"打开浏览器失败: {e}")

def monitor_server(server_process):
    """监控服务器进程"""
    print("\n" + "="*60)
    print("StockAnalysis AI 服务器运行中")
    print("="*60)
    print("\n按 Ctrl+C 停止服务器")
    
    try:
        # 实时输出服务器日志
        for line in iter(server_process.stdout.readline, ''):
            if line:
                print(f"[SERVER] {line.strip()}")
                
    except KeyboardInterrupt:
        print("\n\n正在停止服务器...")
        
    finally:
        # 停止服务器
        if server_process:
            server_process.terminate()
            server_process.wait()
            print("服务器已停止")

def main():
    """主函数"""
    print("="*60)
    print("StockAnalysis AI 应用程序启动")
    print("="*60)
    
    # 检查依赖
    missing_deps = check_dependencies()
    
    if missing_deps:
        if not install_dependencies(missing_deps):
            print("\n❌ 依赖安装失败，请手动安装:")
            print(f"pip install {' '.join(missing_deps)}")
            return 1
    
    # 初始化数据库
    if not initialize_database():
        print("⚠️ 数据库初始化失败，继续启动...")
    
    # 启动服务器
    server_process = start_server()
    if not server_process:
        print("\n❌ 服务器启动失败")
        return 1
    
    # 打开浏览器
    open_browser()
    
    # 监控服务器
    try:
        monitor_server(server_process)
    except Exception as e:
        print(f"服务器监控错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n应用程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 应用程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)