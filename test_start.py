#!/usr/bin/env python3
"""
测试启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入"""
    print("测试模块导入...")
    
    modules = [
        ("backend.config", "配置模块"),
        ("backend.core.gpu_utils", "GPU工具"),
        ("backend.core.model", "模型模块"),
        ("backend.core.engine", "训练引擎"),
        ("backend.api.gpu_router", "GPU路由"),
    ]
    
    for module_path, description in modules:
        try:
            __import__(module_path)
            print(f"[OK] {description}: 导入成功")
        except Exception as e:
            print(f"[ERROR] {description}: 导入失败 - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_gpu_detection():
    """测试GPU检测"""
    print("\n测试GPU检测...")
    
    try:
        from backend.core.gpu_utils import get_gpu_manager
        
        gpu_manager = get_gpu_manager()
        print(f"设备: {gpu_manager.device}")
        print(f"GPU可用: {gpu_manager.is_gpu_available()}")
        
        if gpu_manager.is_gpu_available():
            print(f"GPU信息: {gpu_manager.gpu_info}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] GPU检测失败: {e}")
        return False

def test_config():
    """测试配置"""
    print("\n测试配置系统...")
    
    try:
        from backend.config import settings
        
        print(f"使用GPU: {settings.USE_GPU}")
        print(f"批次大小: {settings.BATCH_SIZE}")
        print(f"训练轮次: {settings.EPOCHS}")
        print(f"学习率: {settings.LEARNING_RATE}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 配置测试失败: {e}")
        return False

def main():
    """主函数"""
    print("StockAnalysis AI 启动测试")
    print("="*50)
    
    tests = [
        ("模块导入", test_imports),
        ("GPU检测", test_gpu_detection),
        ("配置系统", test_config),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        if test_func():
            print(f"[OK] {test_name}: 通过")
        else:
            print(f"[ERROR] {test_name}: 失败")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("[SUCCESS] 所有测试通过!")
        print("\n可以启动服务器:")
        print("  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
    else:
        print("[ERROR] 部分测试失败")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())