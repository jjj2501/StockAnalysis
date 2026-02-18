#!/usr/bin/env python3
"""测试Python环境和基本依赖"""

import sys
import subprocess
import os

print("=" * 60)
print("StockAnalysis AI Environment Test")
print("=" * 60)

# 检查Python版本
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")

# 检查基本模块
modules_to_check = [
    "fastapi",
    "uvicorn", 
    "sqlalchemy",
    "pydantic",
    "torch",
    "numpy",
    "pandas",
    "requests"
]

print("\nChecking dependencies:")
for module in modules_to_check:
    try:
        __import__(module)
        print(f"  OK {module}")
    except ImportError as e:
        print(f"  MISSING {module}: {e}")

# 检查GPU支持
print("\nChecking GPU support:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  OK CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  WARNING CUDA not available (using CPU mode)")
except Exception as e:
    print(f"  ERROR Torch import failed: {e}")

# 检查项目结构
print("\nChecking project structure:")
required_dirs = ["backend", "frontend", "scripts", "docs"]
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"  OK {dir_name}/")
    else:
        print(f"  MISSING {dir_name}/")

print("\n" + "=" * 60)
print("Environment test completed")
print("=" * 60)