from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

from backend.api.enhanced_router import router as main_router
from backend.api.data_router import router as data_router
from backend.api.gpu_router import router as gpu_router
from backend.auth.routers.auth import router as auth_router
from backend.auth.routers.users import router as users_router
from backend.auth.middleware import (
    AuthenticationMiddleware, 
    RateLimitMiddleware, 
    SecurityHeadersMiddleware,
    AuditMiddleware
)
from backend.auth.database import init_db
from backend.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="StockAnalysis AI",
    description="基于 Transformers + LSTM 与 LLM 的智能投资分析工具",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
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

# API 路由
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(main_router, prefix="/api")
app.include_router(data_router, prefix="/api/data", tags=["Market Data Management"])
app.include_router(gpu_router, prefix="/api")

from backend.core.scheduler import scheduler
import asyncio

# === 生命周期管理 ===
@app.on_event("startup")
async def startup_event():
    # 启动后台守护任务更新数据
    asyncio.create_task(scheduler.start_background_loop())
    logging.info("AlphaPulse 后台数据刷新守护进程已启动.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.stop()
    logging.info("AlphaPulse 后台数据刷新守护进程已关闭.")

# 前端托管
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
static_dir = os.path.join(project_root, "frontend")

if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
