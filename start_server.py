#!/usr/bin/env python3
"""
StockAnalysis AI 服务器启动脚本
自动检查端口占用并启动服务器
"""

import subprocess
import sys
import time
import socket
import os
from datetime import datetime

def check_port_in_use(port=8080):
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('localhost', port)) == 0
    except:
        return False

def get_process_using_port(port=8080):
    """获取占用端口的进程信息（Windows）"""
    try:
        # Windows系统使用netstat命令
        result = subprocess.run(
            ['netstat', '-ano', '-p', 'tcp'],
            capture_output=True,
            text=True,
            encoding='gbk'  # Windows中文系统编码
        )
        
        lines = result.stdout.split('\n')
        for line in lines:
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.strip().split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    # 获取进程名
                    try:
                        task_result = subprocess.run(
                            ['tasklist', '/FI', f'PID eq {pid}'],
                            capture_output=True,
                            text=True,
                            encoding='gbk'
                        )
                        return pid, task_result.stdout
                    except:
                        return pid, None
    except Exception as e:
        print(f"获取进程信息时出错: {e}")
    
    return None, None

def kill_process(pid):
    """终止进程"""
    try:
        subprocess.run(['taskkill', '/F', '/PID', pid], check=True)
        print(f"已终止进程 PID: {pid}")
        time.sleep(2)  # 等待进程完全终止
        return True
    except Exception as e:
        print(f"终止进程失败: {e}")
        return False

def start_server():
    """启动服务器"""
    print("=" * 50)
    print("StockAnalysis AI 系统启动")
    print("=" * 50)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查Python
    try:
        python_version = subprocess.run(
            [sys.executable, '--version'],
            capture_output=True,
            text=True
        )
        print(f"Python版本: {python_version.stdout.strip()}")
    except:
        print("错误: 无法找到Python，请确保Python已安装并添加到PATH")
        input("按Enter键退出...")
        return False
    
    # 检查端口
    port = 8080
    if check_port_in_use(port):
        print(f"端口 {port} 已被占用")
        
        pid, process_info = get_process_using_port(port)
        if pid:
            print(f"占用进程 PID: {pid}")
            if process_info:
                print("进程信息:")
                print(process_info)
            
            print()
            choice = input("是否终止该进程并启动新服务器？(y/N): ").strip().lower()
            if choice == 'y':
                if kill_process(pid):
                    print("进程已终止")
                else:
                    print("无法终止进程，请手动处理")
                    input("按Enter键退出...")
                    return False
            else:
                print("取消启动")
                input("按Enter键退出...")
                return False
        else:
            print("无法获取占用进程信息，请手动检查")
            input("按Enter键退出...")
            return False
    
    # 启动服务器
    print()
    print("正在启动StockAnalysis AI服务器...")
    print()
    print("访问地址:")
    print(f"  登录页面: http://localhost:{port}/frontend/login.html")
    print(f"  主仪表盘: http://localhost:{port}/frontend/index.html")
    print(f"  注册页面: http://localhost:{port}/frontend/register.html")
    print()
    print("测试账户:")
    print("  用户名: testuser")
    print("  密码: password123")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)
    print()
    
    try:
        # 启动服务器
        server_process = subprocess.Popen(
            [sys.executable, 'test_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出服务器日志
        print("服务器输出:")
        print("-" * 30)
        for line in server_process.stdout:
            print(line, end='')
            
    except KeyboardInterrupt:
        print("\n\n正在停止服务器...")
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()
        print("服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        return False
    
    return True

def create_shortcut():
    """创建快捷方式（可选）"""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "StockAnalysis AI.lnk")
        
        target = os.path.abspath(sys.executable)
        wDir = os.path.dirname(os.path.abspath(__file__))
        icon = os.path.join(wDir, "test_server.py")  # 使用Python文件作为图标
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = target
        shortcut.Arguments = f'"{os.path.join(wDir, "start_server.py")}"'
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = icon
        shortcut.save()
        
        print(f"已在桌面创建快捷方式: {shortcut_path}")
        return True
    except Exception as e:
        print(f"创建快捷方式失败: {e}")
        return False

if __name__ == "__main__":
    # 检查是否要创建快捷方式
    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        create_shortcut()
        sys.exit(0)
    
    # 启动服务器
    success = start_server()
    
    if not success:
        print("\n启动失败，请检查以上错误信息")
        input("按Enter键退出...")
        sys.exit(1)