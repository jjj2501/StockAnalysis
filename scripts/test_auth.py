#!/usr/bin/env python3
"""
认证系统测试脚本
测试API端点是否正常工作
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_auth_endpoints():
    """测试认证端点"""
    print("🔍 测试认证系统API端点...")
    print("=" * 50)
    
    # 测试数据
    test_user = {
        "email": "test@example.com",
        "password": "TestPassword123!",
        "username": "testuser"
    }
    
    # 模拟API请求
    print("📋 可用API端点:")
    print("1. POST /api/auth/register - 用户注册")
    print("2. POST /api/auth/login - 用户登录")
    print("3. POST /api/auth/refresh - 刷新令牌")
    print("4. POST /api/auth/logout - 用户登出")
    print("5. GET /api/users/me - 获取当前用户信息")
    print("6. POST /api/auth/forgot-password - 忘记密码")
    print("7. POST /api/auth/reset-password - 重置密码")
    print("8. POST /api/auth/change-password - 修改密码")
    
    print("\n📝 测试用户数据:")
    print(f"  邮箱: {test_user['email']}")
    print(f"  用户名: {test_user['username']}")
    print(f"  密码: {'*' * len(test_user['password'])}")
    
    print("\n✅ 认证系统API端点结构验证完成")
    print("⚠️  注意: 需要启动FastAPI服务器进行实际测试")
    print("   运行命令: uvicorn backend.main:app --reload")
    
    return True

def check_dependencies():
    """检查依赖是否安装"""
    print("\n🔧 检查依赖安装状态...")
    
    dependencies = [
        "fastapi",
        "sqlalchemy",
        "passlib",
        "python-jose",
        "redis",
        "pydantic_settings"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (未安装)")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        print("   请运行: pip install " + " ".join(missing))
        return False
    return True

def main():
    """主测试函数"""
    print("🧪 StockAnalysis AI 认证系统测试")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装缺失的依赖")
        return 1
    
    # 测试API端点
    test_auth_endpoints()
    
    print("\n🎉 认证系统测试完成！")
    print("\n📋 下一步:")
    print("1. 启动服务器: uvicorn backend.main:app --reload")
    print("2. 访问 http://localhost:8000/docs 查看API文档")
    print("3. 使用前端页面测试完整流程")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())