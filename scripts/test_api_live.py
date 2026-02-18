#!/usr/bin/env python3
"""
实时API测试脚本
测试正在运行的服务器
"""

import sys
import os
import json
import requests
import time

def test_server_connection():
    """测试服务器连接"""
    print("测试服务器连接...")
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("[SUCCESS] 服务器连接成功")
            return True
        else:
            print(f"[ERROR] 服务器返回状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] 无法连接到服务器，请确保服务器正在运行")
        return False
    except Exception as e:
        print(f"[ERROR] 连接测试失败: {e}")
        return False

def test_register():
    """测试用户注册"""
    print("\n测试用户注册...")
    
    test_user = {
        "email": f"test_{int(time.time())}@example.com",
        "password": "TestPassword123!",
        "username": f"testuser_{int(time.time())}"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/auth/register",
            json=test_user,
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:200]}...")
        
        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS] 用户注册成功: {data.get('email')}")
            return test_user
        else:
            print("[ERROR] 用户注册失败")
            return None
            
    except Exception as e:
        print(f"[ERROR] 注册测试失败: {e}")
        return None

def test_login(user_data):
    """测试用户登录"""
    print("\n测试用户登录...")
    
    login_data = {
        "email": user_data["email"],
        "password": user_data["password"]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/auth/login",
            data=login_data,
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            refresh_token = data.get("refresh_token")
            
            if access_token and refresh_token:
                print(f"[SUCCESS] 用户登录成功")
                print(f"访问令牌: {access_token[:50]}...")
                print(f"刷新令牌: {refresh_token[:50]}...")
                return data
            else:
                print("[ERROR] 登录响应中缺少令牌")
                return None
        else:
            print(f"[ERROR] 登录失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"[ERROR] 登录测试失败: {e}")
        return None

def test_get_current_user(token_data):
    """测试获取当前用户信息"""
    print("\n测试获取当前用户信息...")
    
    access_token = token_data.get("access_token")
    if not access_token:
        print("[ERROR] 没有访问令牌")
        return False
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get(
            "http://localhost:8000/api/users/me",
            headers=headers,
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS] 获取用户信息成功")
            print(f"用户: {data.get('email')} ({data.get('username')})")
            return True
        else:
            print(f"[ERROR] 获取用户信息失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] 获取用户信息测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("StockAnalysis AI 认证系统实时测试")
    print("="*60)
    
    # 测试服务器连接
    if not test_server_connection():
        print("\n[ERROR] 服务器连接测试失败，请先启动服务器")
        print("启动命令: python scripts/start_server.py")
        return 1
    
    print("\n" + "="*60)
    print("开始API端点测试")
    print("="*60)
    
    # 测试注册
    user_data = test_register()
    if not user_data:
        print("\n[ERROR] 注册测试失败，停止后续测试")
        return 1
    
    # 等待一下，确保注册完成
    time.sleep(1)
    
    # 测试登录
    token_data = test_login(user_data)
    if not token_data:
        print("\n[ERROR] 登录测试失败，停止后续测试")
        return 1
    
    # 测试获取用户信息
    if not test_get_current_user(token_data):
        print("\n[ERROR] 获取用户信息测试失败")
        return 1
    
    print("\n" + "="*60)
    print("[SUCCESS] 所有基本API测试通过！")
    print("="*60)
    print("\n下一步:")
    print("1. 访问 http://localhost:8000 查看前端")
    print("2. 访问 http://localhost:8000/docs 查看完整API文档")
    print("3. 运行完整测试: python scripts/test_complete_auth.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())