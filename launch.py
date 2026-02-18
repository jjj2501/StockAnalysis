#!/usr/bin/env python3
"""
启动StockAnalysis AI服务器
"""

import sys
import os
import subprocess
import time
import webbrowser

def start_server():
    """启动服务器"""
    print("启动StockAnalysis AI服务器...")
    
    # 构建命令
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "backend.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--log-level", "info"
    ]
    
    print(f"命令: {' '.join(cmd)}")
    
    # 启动服务器
    process = subprocess.Popen(
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
    
    return process

def check_server():
    """检查服务器是否运行"""
    import requests
    
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

def open_browser():
    """打开浏览器"""
    print("\n在浏览器中打开:")
    print("  前端界面: http://127.0.0.1:8000")
    print("  API文档: http://127.0.0.1:8000/docs")
    print("  GPU状态: http://127.0.0.1:8000/api/gpu/status")
    
    try:
        webbrowser.open("http://127.0.0.1:8000")
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:8000/docs")
    except Exception as e:
        print(f"打开浏览器失败: {e}")

def main():
    """主函数"""
    print("="*60)
    print("StockAnalysis AI 启动")
    print("="*60)
    
    # 启动服务器
    server_process = start_server()
    
    # 检查服务器
    if check_server():
        print("✅ 服务器启动成功!")
        
        # 打开浏览器
        open_browser()
        
        print("\n" + "="*60)
        print("服务器运行中...")
        print("按 Ctrl+C 停止服务器")
        print("="*60)
        
        try:
            # 输出服务器日志
            for line in iter(server_process.stdout.readline, ''):
                if line:
                    print(f"[SERVER] {line.strip()}")
        except KeyboardInterrupt:
            print("\n正在停止服务器...")
        finally:
            server_process.terminate()
            server_process.wait()
            print("服务器已停止")
            
    else:
        print("❌ 服务器启动失败")
        if server_process:
            server_process.terminate()
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n启动被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)