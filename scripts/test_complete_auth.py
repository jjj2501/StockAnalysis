#!/usr/bin/env python3
"""
完整认证流程测试脚本
测试用户注册、登录、令牌刷新、登出等完整流程
"""

import sys
import os
import json
import requests
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AuthTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.test_user = {
            "email": f"test_{int(time.time())}@example.com",
            "password": "TestPassword123!",
            "username": f"testuser_{int(time.time())}"
        }
        self.access_token = None
        self.refresh_token = None
        
    def print_step(self, step_num, description):
        """打印测试步骤"""
        print(f"\n{'='*60}")
        print(f"步骤 {step_num}: {description}")
        print(f"{'='*60}")
    
    def test_register(self):
        """测试用户注册"""
        self.print_step(1, "用户注册")
        
        url = f"{self.api_url}/auth/register"
        response = requests.post(url, json=self.test_user)
        
        print(f"请求URL: {url}")
        print(f"请求数据: {json.dumps(self.test_user, indent=2)}")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据: {json.dumps(data, indent=2)}")
            print("✅ 用户注册成功")
            return True
        else:
            print(f"响应数据: {response.text}")
            print("❌ 用户注册失败")
            return False
    
    def test_login(self):
        """测试用户登录"""
        self.print_step(2, "用户登录")
        
        url = f"{self.api_url}/auth/login"
        login_data = {
            "email": self.test_user["email"],
            "password": self.test_user["password"]
        }
        
        response = requests.post(url, data=login_data)
        
        print(f"请求URL: {url}")
        print(f"请求数据: {json.dumps(login_data, indent=2)}")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据: {json.dumps(data, indent=2)}")
            
            # 保存令牌
            self.access_token = data.get("access_token")
            self.refresh_token = data.get("refresh_token")
            
            if self.access_token and self.refresh_token:
                print("✅ 用户登录成功，获取到令牌")
                return True
            else:
                print("❌ 登录响应中缺少令牌")
                return False
        else:
            print(f"响应数据: {response.text}")
            print("❌ 用户登录失败")
            return False
    
    def test_get_current_user(self):
        """测试获取当前用户信息"""
        self.print_step(3, "获取当前用户信息")
        
        if not self.access_token:
            print("❌ 没有访问令牌，无法测试")
            return False
        
        url = f"{self.api_url}/users/me"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        response = requests.get(url, headers=headers)
        
        print(f"请求URL: {url}")
        print(f"请求头: {json.dumps(headers, indent=2)}")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据: {json.dumps(data, indent=2)}")
            print("✅ 成功获取当前用户信息")
            return True
        else:
            print(f"响应数据: {response.text}")
            print("❌ 获取用户信息失败")
            return False
    
    def test_refresh_token(self):
        """测试刷新令牌"""
        self.print_step(4, "刷新访问令牌")
        
        if not self.refresh_token:
            print("❌ 没有刷新令牌，无法测试")
            return False
        
        url = f"{self.api_url}/auth/refresh"
        data = {"refresh_token": self.refresh_token}
        
        response = requests.post(url, json=data)
        
        print(f"请求URL: {url}")
        print(f"请求数据: {json.dumps(data, indent=2)}")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据: {json.dumps(data, indent=2)}")
            
            # 更新访问令牌
            new_access_token = data.get("access_token")
            if new_access_token:
                self.access_token = new_access_token
                print("✅ 令牌刷新成功")
                return True
            else:
                print("❌ 刷新响应中缺少新访问令牌")
                return False
        else:
            print(f"响应数据: {response.text}")
            print("❌ 令牌刷新失败")
            return False
    
    def test_logout(self):
        """测试用户登出"""
        self.print_step(5, "用户登出")
        
        if not self.refresh_token:
            print("❌ 没有刷新令牌，无法测试")
            return False
        
        url = f"{self.api_url}/auth/logout"
        headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
        data = {"refresh_token": self.refresh_token}
        
        response = requests.post(url, headers=headers, json=data)
        
        print(f"请求URL: {url}")
        print(f"请求头: {json.dumps(headers, indent=2)}")
        print(f"请求数据: {json.dumps(data, indent=2)}")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应数据: {json.dumps(data, indent=2)}")
            print("✅ 用户登出成功")
            return True
        else:
            print(f"响应数据: {response.text}")
            print("❌ 用户登出失败")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 StockAnalysis AI 完整认证流程测试")
        print("="*60)
        
        tests_passed = 0
        total_tests = 5
        
        # 测试1: 注册
        if self.test_register():
            tests_passed += 1
        
        # 等待一下，确保注册完成
        time.sleep(1)
        
        # 测试2: 登录
        if self.test_login():
            tests_passed += 1
        
        # 测试3: 获取用户信息
        if self.test_get_current_user():
            tests_passed += 1
        
        # 测试4: 刷新令牌
        if self.test_refresh_token():
            tests_passed += 1
        
        # 测试5: 登出
        if self.test_logout():
            tests_passed += 1
        
        # 测试结果汇总
        print(f"\n{'='*60}")
        print("测试结果汇总")
        print(f"{'='*60}")
        print(f"✅ 通过的测试: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("🎉 所有认证流程测试通过！")
            return True
        else:
            print(f"⚠️  部分测试失败 ({total_tests - tests_passed}个)")
            return False

def main():
    """主函数"""
    tester = AuthTester()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 无法连接到服务器: {tester.base_url}")
        print("请确保FastAPI服务器正在运行:")
        print("  1. 启动服务器: uvicorn backend.main:app --reload")
        print("  2. 然后重新运行此测试")
        return 1
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())