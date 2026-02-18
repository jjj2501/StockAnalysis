#!/usr/bin/env python3
"""
前端认证流程测试
模拟用户在前端的完整操作流程
"""

import sys
import os
import json
import requests
import time
import webbrowser
from datetime import datetime

class FrontendTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_user = {
            "email": f"frontend_test_{int(time.time())}@example.com",
            "password": "FrontendTest123!",
            "username": f"frontend_user_{int(time.time())}"
        }
        self.access_token = None
        self.refresh_token = None
        
    def print_step(self, step_num, description):
        """打印测试步骤"""
        print(f"\n{'='*60}")
        print(f"步骤 {step_num}: {description}")
        print(f"{'='*60}")
    
    def test_frontend_pages(self):
        """测试前端页面可访问性"""
        self.print_step(1, "测试前端页面可访问性")
        
        pages = [
            ("/", "首页"),
            ("/login.html", "登录页面"),
            ("/register.html", "注册页面"),
            ("/dashboard.html", "用户仪表板"),
        ]
        
        all_accessible = True
        for page, name in pages:
            try:
                response = requests.get(f"{self.base_url}{page}", timeout=5)
                status = "✓" if response.status_code == 200 else "✗"
                print(f"  {status} {name}: HTTP {response.status_code}")
                
                if response.status_code != 200:
                    all_accessible = False
                    
            except Exception as e:
                print(f"  ✗ {name}: 无法访问 - {e}")
                all_accessible = False
        
        return all_accessible
    
    def test_registration_flow(self):
        """测试注册流程"""
        self.print_step(2, "测试用户注册流程")
        
        print(f"测试用户: {self.test_user['email']}")
        
        # 模拟注册API调用
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/register",
                json=self.test_user,
                timeout=10
            )
            
            print(f"注册API响应: HTTP {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"[SUCCESS] 注册成功: {data.get('email')}")
                return True
            else:
                print(f"[ERROR] 注册失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 注册API调用失败: {e}")
            return False
    
    def test_login_flow(self):
        """测试登录流程"""
        self.print_step(3, "测试用户登录流程")
        
        login_data = {
            "email": self.test_user["email"],
            "password": self.test_user["password"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                data=login_data,
                timeout=10
            )
            
            print(f"登录API响应: HTTP {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                
                if self.access_token and self.refresh_token:
                    print(f"[SUCCESS] 登录成功")
                    print(f"  访问令牌获取成功")
                    print(f"  刷新令牌获取成功")
                    print(f"  用户: {data.get('user', {}).get('username')}")
                    return True
                else:
                    print("[ERROR] 登录响应中缺少令牌")
                    return False
            else:
                print(f"[ERROR] 登录失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 登录API调用失败: {e}")
            return False
    
    def test_dashboard_access(self):
        """测试仪表板访问"""
        self.print_step(4, "测试用户仪表板访问")
        
        if not self.access_token:
            print("[ERROR] 没有访问令牌，无法测试仪表板")
            return False
        
        # 测试获取用户信息
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(
                f"{self.base_url}/api/users/me",
                headers=headers,
                timeout=10
            )
            
            print(f"用户信息API响应: HTTP {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"[SUCCESS] 仪表板访问成功")
                print(f"  用户ID: {data.get('id')}")
                print(f"  用户名: {data.get('username')}")
                print(f"  邮箱: {data.get('email')}")
                print(f"  角色: {data.get('role')}")
                return True
            else:
                print(f"[ERROR] 仪表板访问失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 仪表板API调用失败: {e}")
            return False
    
    def test_logout_flow(self):
        """测试登出流程"""
        self.print_step(5, "测试用户登出流程")
        
        if not self.refresh_token:
            print("[ERROR] 没有刷新令牌，无法测试登出")
            return False
        
        headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
        data = {"refresh_token": self.refresh_token}
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/logout",
                headers=headers,
                json=data,
                timeout=10
            )
            
            print(f"登出API响应: HTTP {response.status_code}")
            
            if response.status_code == 200:
                print("[SUCCESS] 登出成功")
                return True
            else:
                print(f"[ERROR] 登出失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 登出API调用失败: {e}")
            return False
    
    def open_browser_for_manual_test(self):
        """打开浏览器进行手动测试"""
        self.print_step(6, "打开浏览器进行手动测试")
        
        print("将在浏览器中打开以下页面进行手动测试:")
        print(f"  1. 首页: {self.base_url}/")
        print(f"  2. 登录页面: {self.base_url}/login.html")
        print(f"  3. 注册页面: {self.base_url}/register.html")
        print(f"  4. API文档: {self.base_url}/docs")
        
        try:
            # 打开首页
            webbrowser.open(f"{self.base_url}/")
            time.sleep(1)
            
            # 打开登录页面
            webbrowser.open(f"{self.base_url}/login.html")
            time.sleep(1)
            
            # 打开API文档
            webbrowser.open(f"{self.base_url}/docs")
            
            print("\n[INFO] 浏览器已打开，请手动测试前端功能")
            print("测试用户信息:")
            print(f"  邮箱: {self.test_user['email']}")
            print(f"  密码: {self.test_user['password']}")
            print(f"  用户名: {self.test_user['username']}")
            
        except Exception as e:
            print(f"[ERROR] 打开浏览器失败: {e}")
            print("请手动访问以上URL进行测试")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 StockAnalysis AI 前端认证流程测试")
        print("="*60)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"服务器地址: {self.base_url}")
        print("="*60)
        
        tests_passed = 0
        total_tests = 5
        
        # 测试1: 前端页面可访问性
        if self.test_frontend_pages():
            tests_passed += 1
        
        # 等待一下
        time.sleep(1)
        
        # 测试2: 注册流程
        if self.test_registration_flow():
            tests_passed += 1
        
        # 等待注册完成
        time.sleep(2)
        
        # 测试3: 登录流程
        if self.test_login_flow():
            tests_passed += 1
        
        # 测试4: 仪表板访问
        if self.test_dashboard_access():
            tests_passed += 1
        
        # 测试5: 登出流程
        if self.test_logout_flow():
            tests_passed += 1
        
        # 测试结果汇总
        print(f"\n{'='*60}")
        print("测试结果汇总")
        print(f"{'='*60}")
        print(f"通过的测试: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("[SUCCESS] 所有前端认证流程测试通过！")
            
            # 询问是否打开浏览器进行手动测试
            print("\n是否要打开浏览器进行手动测试？")
            response = input("输入 'y' 确认，其他键跳过: ")
            if response.lower() == 'y':
                self.open_browser_for_manual_test()
            
            return True
        else:
            print(f"[WARNING] 部分测试失败 ({total_tests - tests_passed}个)")
            print("\n建议:")
            print("1. 确保服务器正在运行")
            print("2. 检查数据库连接")
            print("3. 查看服务器日志")
            return False

def main():
    """主函数"""
    tester = FrontendTester()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())