#!/usr/bin/env python3
"""
GPU训练功能测试脚本
测试StockAnalysis AI的GPU加速训练功能
"""

import sys
import os
import json
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gpu_detection():
    """测试GPU检测功能"""
    print("="*60)
    print("测试GPU检测功能")
    print("="*60)
    
    try:
        import torch
        from backend.core.gpu_utils import get_gpu_manager, device_info
        
        gpu_manager = get_gpu_manager()
        info = device_info()
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"设备: {info['device']}")
        print(f"GPU可用: {info['gpu_available']}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print(f"配置 - 使用GPU: {info['config']['use_gpu']}")
        print(f"配置 - GPU设备ID: {info['config']['gpu_device_id']}")
        
        return True
        
    except Exception as e:
        print(f"GPU检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_memory_management():
    """测试GPU内存管理"""
    print("\n" + "="*60)
    print("测试GPU内存管理")
    print("="*60)
    
    try:
        from backend.core.gpu_utils import get_gpu_manager
        
        gpu_manager = get_gpu_manager()
        
        print("获取GPU内存信息...")
        memory_info = gpu_manager.get_memory_info()
        
        if gpu_manager.is_gpu_available():
            print(f"已分配内存: {memory_info.get('allocated', 'N/A'):.1f} MB")
            print(f"保留内存: {memory_info.get('reserved', 'N/A'):.1f} MB")
            print(f"最大已分配: {memory_info.get('max_allocated', 'N/A'):.1f} MB")
            print(f"最大保留: {memory_info.get('max_reserved', 'N/A'):.1f} MB")
        else:
            print("GPU不可用，跳过内存测试")
        
        print("\n测试缓存清除...")
        gpu_manager.clear_cache()
        print("缓存清除完成")
        
        return True
        
    except Exception as e:
        print(f"GPU内存管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_gpu_transfer():
    """测试模型GPU转移"""
    print("\n" + "="*60)
    print("测试模型GPU转移")
    print("="*60)
    
    try:
        import torch
        from backend.core.model import HybridModel
        from backend.core.gpu_utils import get_gpu_manager
        
        gpu_manager = get_gpu_manager()
        
        # 创建模型
        print("创建HybridModel...")
        model = HybridModel(
            input_dim=12,
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        )
        
        print(f"模型初始设备: 未指定")
        
        # 优化模型（移动到GPU如果可用）
        print("优化模型（GPU加速）...")
        model = gpu_manager.optimize_for_training(model, batch_size=32)
        
        print(f"模型最终设备: {next(model.parameters()).device}")
        
        # 测试前向传播
        print("测试前向传播...")
        test_input = torch.randn(2, 60, 12).to(next(model.parameters()).device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出设备: {output.device}")
        
        return True
        
    except Exception as e:
        print(f"模型GPU转移测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_engine():
    """测试训练引擎"""
    print("\n" + "="*60)
    print("测试训练引擎")
    print("="*60)
    
    try:
        from backend.core.engine import StockEngine
        from backend.config import settings
        
        print("初始化训练引擎...")
        engine = StockEngine(model_dir="backend/models/test_gpu")
        
        print(f"训练引擎设备: {engine.device}")
        print(f"GPU可用: {engine.gpu_manager.is_gpu_available()}")
        
        if engine.gpu_manager.is_gpu_available():
            print(f"GPU信息: {engine.gpu_manager.gpu_info.get('device_name', 'Unknown')}")
        
        print(f"批次大小: {engine.batch_size}")
        print(f"训练轮次: {engine.epochs}")
        print(f"学习率: {engine.learning_rate}")
        print(f"隐藏层维度: {engine.hidden_dim}")
        print(f"序列长度: {engine.seq_length}")
        
        # 测试模型创建
        print("\n测试模型创建...")
        model = engine._get_model()
        print(f"模型设备: {next(model.parameters()).device}")
        
        # 测试数据准备（模拟）
        print("\n测试数据准备...")
        import numpy as np
        
        # 创建模拟数据
        seq_length = engine.seq_length
        batch_size = 32
        input_dim = 12
        
        X_train = np.random.randn(batch_size * 10, seq_length, input_dim).astype(np.float32)
        y_train = np.random.randn(batch_size * 10).astype(np.float32)
        
        print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
        
        # 测试数据加载器创建
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=engine.batch_size,
            shuffle=True,
            pin_memory=engine.gpu_manager.is_gpu_available()
        )
        
        print(f"数据加载器创建成功")
        print(f"批次数量: {len(train_loader)}")
        print(f"pin_memory: {train_loader.pin_memory}")
        
        return True
        
    except Exception as e:
        print(f"训练引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_system():
    """测试配置系统"""
    print("\n" + "="*60)
    print("测试配置系统")
    print("="*60)
    
    try:
        from backend.config import settings
        
        print("当前GPU/训练配置:")
        print(f"  USE_GPU: {settings.USE_GPU}")
        print(f"  GPU_DEVICE_ID: {settings.GPU_DEVICE_ID}")
        print(f"  BATCH_SIZE: {settings.BATCH_SIZE}")
        print(f"  EPOCHS: {settings.EPOCHS}")
        print(f"  LEARNING_RATE: {settings.LEARNING_RATE}")
        print(f"  MODEL_HIDDEN_DIM: {settings.MODEL_HIDDEN_DIM}")
        print(f"  MODEL_NUM_LAYERS: {settings.MODEL_NUM_LAYERS}")
        print(f"  SEQUENCE_LENGTH: {settings.SEQUENCE_LENGTH}")
        
        # 测试环境变量覆盖
        print("\n测试环境变量覆盖:")
        os.environ["USE_GPU"] = "False"
        os.environ["BATCH_SIZE"] = "64"
        
        # 重新导入设置以应用环境变量
        import importlib
        import backend.config
        importlib.reload(backend.config)
        from backend.config import settings as new_settings
        
        print(f"  覆盖后 USE_GPU: {new_settings.USE_GPU}")
        print(f"  覆盖后 BATCH_SIZE: {new_settings.BATCH_SIZE}")
        
        # 恢复环境变量
        del os.environ["USE_GPU"]
        del os.environ["BATCH_SIZE"]
        
        return True
        
    except Exception as e:
        print(f"配置系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """测试API端点（如果服务器运行）"""
    print("\n" + "="*60)
    print("测试API端点")
    print("="*60)
    
    try:
        import requests
        
        # 检查服务器是否运行
        try:
            response = requests.get("http://localhost:8000/docs", timeout=5)
            if response.status_code != 200:
                print("服务器未运行，跳过API测试")
                print("启动服务器: python -m uvicorn backend.main:app --reload")
                return True  # 不是错误，只是跳过
        except:
            print("服务器未运行，跳过API测试")
            print("启动服务器: python -m uvicorn backend.main:app --reload")
            return True
        
        print("服务器正在运行，测试GPU API端点...")
        
        # 测试GPU状态端点
        print("\n1. 测试 /api/gpu/status")
        response = requests.get("http://localhost:8000/api/gpu/status", timeout=10)
        print(f"  状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  设备: {data.get('device')}")
            print(f"  GPU可用: {data.get('gpu_available')}")
        
        # 测试GPU设备端点
        print("\n2. 测试 /api/gpu/devices")
        response = requests.get("http://localhost:8000/api/gpu/devices", timeout=10)
        print(f"  状态码: {response.status_code}")
        
        # 测试配置端点
        print("\n3. 测试 /api/gpu/config/current")
        response = requests.get("http://localhost:8000/api/gpu/config/current", timeout=10)
        print(f"  状态码: {response.status_code}")
        if response.status_code == 200:
            config = response.json()
            print(f"  使用GPU: {config.get('use_gpu')}")
            print(f"  批次大小: {config.get('batch_size')}")
        
        return True
        
    except Exception as e:
        print(f"API端点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("StockAnalysis AI GPU训练功能测试")
    print("="*60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("GPU检测", test_gpu_detection),
        ("GPU内存管理", test_gpu_memory_management),
        ("模型GPU转移", test_model_gpu_transfer),
        ("训练引擎", test_training_engine),
        ("配置系统", test_configuration_system),
        ("API端点", test_api_endpoints),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"[OK] {test_name}: 通过")
                passed_tests += 1
            else:
                print(f"[ERROR] {test_name}: 失败")
        except Exception as e:
            print(f"[ERROR] {test_name}: 异常 - {e}")
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] 所有GPU训练功能测试通过！")
        print("\n下一步:")
        print("1. 启动服务器: python -m uvicorn backend.main:app --reload")
        print("2. 访问 http://localhost:8000/docs 查看GPU API文档")
        print("3. 使用前端界面测试完整训练流程")
        return 0
    else:
        print(f"\n[WARNING] 部分测试失败 ({total_tests - passed_tests}个)")
        print("\n建议:")
        print("1. 检查PyTorch安装是否正确")
        print("2. 验证CUDA驱动（如果使用GPU）")
        print("3. 查看错误日志")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)