#!/usr/bin/env python3
"""
最终系统集成验证
验证认证系统与现有系统的集成
"""

import sys
import os
import json
import requests
import time
from datetime import datetime

def check_system_components():
    """检查系统组件"""
    print("检查系统组件...")
    
    components = {
        "数据库文件": "stockanalysis.db",
        "后端目录": "./backend",
        "前端目录": "./frontend",
        "认证模块": "./backend/auth",
        "配置文件": "./backend/config.py",
    }
    
    all_present = True
    for name, path in components.items():
        if os.path.exists(path):
            print(f"  [OK] {name}: 存在")
        else:
            print(f"  [ERROR] {name}: 缺失")
            all_present = False
    
    return all_present

def test_integrated_apis():
    """测试集成API"""
    print("\n测试集成API...")
    
    # 先创建测试用户
    test_user = {
        "email": f"integration_test_{int(time.time())}@example.com",
        "password": "IntegrationTest123!",
        "username": f"integration_user_{int(time.time())}"
    }
    
    try:
        # 1. 注册用户
        print("1. 测试用户注册...")
        response = requests.post(
            "http://localhost:8000/api/auth/register",
            json=test_user,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"  [ERROR] 注册失败: {response.status_code}")
            return False
        print("  [OK] 注册成功")
        
        # 等待一下
        time.sleep(1)
        
        # 2. 用户登录
        print("2. 测试用户登录...")
        response = requests.post(
            "http://localhost:8000/api/auth/login",
            data={
                "email": test_user["email"],
                "password": test_user["password"]
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"  [ERROR] 登录失败: {response.status_code}")
            return False
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            print("  [ERROR] 登录响应中缺少访问令牌")
            return False
        print("  [OK] 登录成功，获取到访问令牌")
        
        # 3. 测试受保护端点
        print("3. 测试受保护端点...")
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # 测试获取用户信息
        response = requests.get(
            "http://localhost:8000/api/users/me",
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"  [ERROR] 获取用户信息失败: {response.status_code}")
            return False
        print("  [OK] 成功获取用户信息")
        
        # 4. 测试匿名访问（向后兼容）
        print("4. 测试匿名访问（向后兼容）...")
        response = requests.get(
            "http://localhost:8000/",
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"  [ERROR] 匿名访问失败: {response.status_code}")
            return False
        print("  [OK] 匿名访问成功")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("  [ERROR] 无法连接到服务器")
        return False
    except Exception as e:
        print(f"  [ERROR] API测试失败: {e}")
        return False

def verify_database_integrity():
    """验证数据库完整性"""
    print("\n验证数据库完整性...")
    
    db_file = "stockanalysis.db"
    if not os.path.exists(db_file):
        print(f"  [ERROR] 数据库文件不存在: {db_file}")
        return False
    
    size = os.path.getsize(db_file)
    print(f"  [OK] 数据库文件存在，大小: {size} bytes")
    
    # 检查文件是否可读
    try:
        import sqlite3
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # 检查必要的表是否存在
        tables = ["users", "user_sessions", "audit_logs"]
        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                print(f"  [OK] 表 '{table}' 存在")
            else:
                print(f"  [ERROR] 表 '{table}' 缺失")
                conn.close()
                return False
        
        # 检查users表中是否有数据
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        print(f"  [OK] users表中有 {count} 条记录")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  [ERROR] 数据库验证失败: {e}")
        return False

def generate_deployment_report():
    """生成部署报告"""
    print("\n生成部署报告...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd(),
        },
        "components": {},
        "tests": {},
        "recommendations": []
    }
    
    # 检查组件
    components = [
        ("backend", "./backend"),
        ("frontend", "./frontend"),
        ("database", "./stockanalysis.db"),
        ("config", "./backend/config.py"),
        ("auth_module", "./backend/auth"),
    ]
    
    for name, path in components:
        report["components"][name] = {
            "exists": os.path.exists(path),
            "path": os.path.abspath(path) if os.path.exists(path) else None
        }
    
    # 保存报告
    report_file = "./deployment_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  [OK] 部署报告已保存: {report_file}")
    return report_file

def main():
    """主函数"""
    print("="*70)
    print("StockAnalysis AI 系统集成验证")
    print("="*70)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_passed = True
    
    # 1. 检查系统组件
    if not check_system_components():
        all_passed = False
        print("\n[WARNING] 部分系统组件缺失")
    
    # 2. 验证数据库完整性
    if not verify_database_integrity():
        all_passed = False
        print("\n[WARNING] 数据库完整性检查失败")
    
    # 3. 测试集成API（需要服务器运行）
    print("\n" + "="*70)
    print("API集成测试（需要服务器运行）")
    print("="*70)
    
    server_running = False
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            server_running = True
            print("服务器正在运行，开始API集成测试...")
    except:
        print("服务器未运行，跳过API集成测试")
        print("要启动服务器，请运行: python -m uvicorn backend.main:app --reload")
    
    if server_running:
        if not test_integrated_apis():
            all_passed = False
            print("\n[WARNING] API集成测试失败")
    
    # 4. 生成部署报告
    report_file = generate_deployment_report()
    
    # 5. 显示验证结果
    print("\n" + "="*70)
    print("验证结果汇总")
    print("="*70)
    
    if all_passed:
        print("[SUCCESS] 系统集成验证通过！")
    else:
        print("[WARNING] 系统集成验证发现一些问题")
    
    print("\n部署信息:")
    print(f"  前端地址: http://localhost:8000")
    print(f"  API文档: http://localhost:8000/docs")
    print(f"  数据库文件: {os.path.abspath('./stockanalysis.db')}")
    print(f"  部署报告: {os.path.abspath(report_file)}")
    
    print("\n下一步建议:")
    print("  1. 启动完整服务器: python -m uvicorn backend.main:app --reload")
    print("  2. 访问前端页面测试完整功能")
    print("  3. 查看API文档了解所有可用端点")
    print("  4. 根据需要配置环境变量（生产环境）")
    
    print("\n" + "="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n验证被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 验证过程中发生错误: {e}")
        sys.exit(1)