#!/usr/bin/env python3
"""
简化服务器启动脚本
仅启动认证系统，避免加载所有依赖
"""

import sys
import os
import uvicorn

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 创建简化的FastAPI应用
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

# 只导入认证相关模块
from backend.auth.routers.auth import router as auth_router
from backend.auth.routers.users import router as users_router
from backend.auth.middleware import (
    AuthenticationMiddleware, 
    RateLimitMiddleware, 
    SecurityHeadersMiddleware,
    AuditMiddleware
)
from backend.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="StockAnalysis AI - 认证系统",
    description="用户认证和管理API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加自定义中间件
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditMiddleware)

# 只包含认证路由
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")

# 前端托管
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
static_dir = os.path.join(project_root, "frontend")

if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    print("[INFO] 启动StockAnalysis AI认证服务器...")
    print(f"[INFO] 文档地址: http://localhost:8000/docs")
    print(f"[INFO] 前端地址: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)