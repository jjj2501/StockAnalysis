#!/usr/bin/env python3
"""
数据库初始化脚本
用于创建认证系统所需的数据库表
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.auth.database import init_db, drop_db
from backend.config import settings

def main():
    """初始化数据库"""
    print(f"正在初始化数据库: {settings.DATABASE_URL}")
    
    try:
        # 创建所有表
        init_db()
        print("[SUCCESS] 数据库表创建成功！")
        
        # 显示数据库文件信息（如果是SQLite）
        if settings.DATABASE_URL.startswith("sqlite"):
            db_file = settings.DATABASE_URL.replace("sqlite:///", "")
            if os.path.exists(db_file):
                size = os.path.getsize(db_file)
                print(f"[INFO] 数据库文件: {db_file}")
                print(f"[INFO] 文件大小: {size} bytes")
        
    except Exception as e:
        print(f"[ERROR] 数据库初始化失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()