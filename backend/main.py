from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.api.router import router
import os

app = FastAPI(title="StockAnalysis AI")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(router, prefix="/api")

# 前端托管
# 获取当前文件所在目录的上级目录 (即 backend)，再同级的 frontend
current_dir = os.path.dirname(os.path.abspath(__file__))
# backend/main.py -> backend -> parent -> frontend
project_root = os.path.dirname(current_dir)
static_dir = os.path.join(project_root, "frontend")

if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # 监听 0.0.0.0 方便局域网访问, 端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
