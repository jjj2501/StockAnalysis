#!/usr/bin/env python3
"""
快速测试脚本
"""

import sys
import os
import subprocess
import time
import requests

def start_server():
    """启动服务器"""
    print("启动认证服务器...")
    
    # 启动服务器进程
    server_process = subprocess.Popen(
        [sys.executable, "scripts/start_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(3)
    
    return server_process

def test_basic_endpoints():
    """测试基本端点"""
    print("\n测试基本端点...")
    
    endpoints = [
        ("/", "前端首页"),
        ("/docs", "API文档"),
        ("/api/auth/register", "注册端点"),
        ("/api/auth/login", "登录端点"),
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            print(f"  {description}: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  {description}: 连接失败")
            return False
        except Exception as e:
            print(f"  {description}: 错误 - {e}")
    
    return True

def main():
    """主函数"""
    print("StockAnalysis AI 快速部署测试")
    print("="*50)
    
    # 启动服务器
    server_process = start_server()
    
    try:
        # 测试端点
        if test_basic_endpoints():
            print("\n[SUCCESS] 基本端点测试通过！")
            print("\n服务器运行中，可以访问:")
            print("  http://localhost:8000 - 前端界面")
            print("  http://localhost:8000/docs - API文档")
            print("\n按Ctrl+C停止服务器")
            
            # 保持服务器运行
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n停止服务器...")
        else:
            print("\n[ERROR] 端点测试失败")
            
    finally:
        # 停止服务器
        if server_process:
            server_process.terminate()
            server_process.wait()
            print("服务器已停止")

if __name__ == "__main__":
    main()