#!/usr/bin/env python3
"""
StockAnalysis AI 服务器运行脚本
简化版本，自动处理端口冲突
"""

import subprocess
import sys
import time
import socket
import os
from datetime import datetime

def check_port(port=8080):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('localhost', port)) == 0
    except:
        return False

def kill_port_process(port=8080):
    """终止占用端口的进程"""
    try:
        # Windows系统
        result = subprocess.run(
            ['netstat', '-ano', '-p', 'tcp'],
            capture_output=True,
            text=True,
            encoding='gbk'
        )
        
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.strip().split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    print(f"发现占用端口 {port} 的进程 PID: {pid}")
                    
                    # 获取进程名
                    try:
                        task_result = subprocess.run(
                            ['tasklist', '/FI', f'PID eq {pid}'],
                            capture_output=True,
                            text=True,
                            encoding='gbk'
                        )
                        print(f"进程信息:\n{task_result.stdout}")
                    except:
                        pass
                    
                    # 终止进程
                    try:
                        subprocess.run(['taskkill', '/F', '/PID', pid], check=True)
                        print(f"已终止进程 PID: {pid}")
                        time.sleep(2)
                        return True
                    except Exception as e:
                        print(f"终止进程失败: {e}")
                        return False
        return False
    except Exception as e:
        print(f"检查端口进程时出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("StockAnalysis AI - 智能股票分析系统")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查Python
    try:
        subprocess.run([sys.executable, '--version'], check=True)
    except:
        print("错误: Python未安装或未添加到PATH")
        print("请安装Python 3.10或更高版本")
        input("按Enter键退出...")
        return
    
    port = 8080
    
    # 检查并处理端口占用
    if check_port(port):
        print(f"端口 {port} 已被占用，正在处理...")
        if kill_port_process(port):
            print("端口已释放")
        else:
            print("无法释放端口，请手动关闭占用程序")
            print("或修改 test_server.py 中的端口号")
            input("按Enter键退出...")
            return
    
    print()
    print("启动服务器...")
    print()
    print("系统访问地址:")
    print(f"  登录页面: http://localhost:{port}/frontend/login.html")
    print(f"  主仪表盘: http://localhost:{port}/frontend/index.html")
    print(f"  注册页面: http://localhost:{port}/frontend/register.html")
    print()
    print("测试账户:")
    print("  用户名: testuser")
    print("  密码: password123")
    print()
    print("API接口:")
    print(f"  健康检查: http://localhost:{port}/api/health")
    print(f"  股票列表: http://localhost:{port}/api/stocks")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    print()
    
    try:
        # 启动服务器
        process = subprocess.Popen(
            [sys.executable, 'test_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 输出服务器日志
        print("服务器日志:")
        print("-" * 40)
        for line in process.stdout:
            print(line, end='')
            
    except KeyboardInterrupt:
        print("\n\n正在停止服务器...")
        process.terminate()
        process.wait()
        print("服务器已停止")
    except Exception as e:
        print(f"服务器错误: {e}")
    finally:
        print()
        print("=" * 60)
        print("StockAnalysis AI 服务器已关闭")
        print("=" * 60)

if __name__ == "__main__":
    main()