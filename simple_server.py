#!/usr/bin/env python3
"""
简化版StockAnalysis AI服务器
用于测试界面和基本功能
"""

import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext

# 创建FastAPI应用
app = FastAPI(
    title="StockAnalysis AI",
    description="简化版智能投资分析工具",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# 数据库连接
DATABASE = "stockanalysis.db"

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'user',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建用户会话表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            refresh_token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # 创建自选股表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            note TEXT,
            UNIQUE(user_id, symbol),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# 初始化数据库
init_db()

# Pydantic模型
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码至少需要8个字符')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    created_at: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class StockSymbol(BaseModel):
    symbol: str
    name: Optional[str] = None
    note: Optional[str] = None

class StockAnalysis(BaseModel):
    symbol: str
    name: str
    current_price: float
    price_change: float
    prediction: str
    confidence: float
    factors: Dict[str, Any]
    history: List[Dict[str, Any]]
    report: str

# 工具函数
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_tokens(user_id: int) -> Dict[str, str]:
    access_token = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(days=7)
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_sessions (user_id, session_token, refresh_token, expires_at) VALUES (?, ?, ?, ?)",
        (user_id, access_token, refresh_token, expires_at)
    )
    conn.commit()
    conn.close()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at.isoformat()
    }

def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT u.* FROM users u JOIN user_sessions s ON u.id = s.user_id WHERE s.session_token = ? AND s.expires_at > ?",
        (token, datetime.now())
    )
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return dict(user)

# 认证路由
@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserCreate):
    conn = get_db()
    cursor = conn.cursor()
    
    # 检查用户名是否已存在
    cursor.execute("SELECT id FROM users WHERE username = ?", (user.username,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 检查邮箱是否已存在
    cursor.execute("SELECT id FROM users WHERE email = ?", (user.email,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="邮箱已存在")
    
    # 创建用户
    hashed_password = hash_password(user.password)
    cursor.execute(
        "INSERT INTO users (username, email, hashed_password, full_name) VALUES (?, ?, ?, ?)",
        (user.username, user.email, hashed_password, user.full_name)
    )
    user_id = cursor.lastrowid
    
    conn.commit()
    
    # 获取创建的用户
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    new_user = cursor.fetchone()
    conn.close()
    
    return dict(new_user)

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    cursor = conn.cursor()
    
    # 查找用户
    cursor.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (form_data.username, form_data.username)
    )
    user = cursor.fetchone()
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    
    if not user["is_active"]:
        raise HTTPException(status_code=400, detail="账户已被禁用")
    
    # 创建令牌
    tokens = create_tokens(user["id"])
    conn.close()
    
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "token_type": "bearer"
    }

@app.post("/api/auth/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        "DELETE FROM user_sessions WHERE user_id = ?",
        (current_user["id"],)
    )
    conn.commit()
    conn.close()
    
    return {"message": "已成功登出"}

@app.get("/api/users/me", response_model=UserResponse)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return current_user

# 自选股路由
@app.get("/api/watchlist")
async def get_watchlist(current_user: Dict[str, Any] = Depends(get_current_user)):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM watchlist WHERE user_id = ? ORDER BY added_at DESC",
        (current_user["id"],)
    )
    watchlist = cursor.fetchall()
    conn.close()
    
    return [dict(item) for item in watchlist]

@app.post("/api/watchlist")
async def add_to_watchlist(
    stock: StockSymbol,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO watchlist (user_id, symbol, name, note) VALUES (?, ?, ?, ?)",
            (current_user["id"], stock.symbol, stock.name, stock.note)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="该股票已在自选股中")
    finally:
        conn.close()
    
    return {"message": "已添加到自选股"}

@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(
    symbol: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?",
        (current_user["id"], symbol)
    )
    conn.commit()
    conn.close()
    
    return {"message": "已从自选股移除"}

# 股票分析路由
@app.get("/api/stock/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # 模拟股票数据
    import random
    import time
    
    base_price = random.uniform(50, 200)
    change = random.uniform(-5, 5)
    prediction = "UP" if random.random() > 0.5 else "DOWN"
    confidence = random.uniform(70, 95)
    
    # 生成历史数据
    history = []
    now = datetime.now()
    for i in range(29, -1, -1):
        date = now - timedelta(days=i)
        open_price = base_price + random.uniform(-3, 3)
        close_price = open_price + random.uniform(-2, 2)
        high_price = max(open_price, close_price) + random.uniform(0, 1)
        low_price = min(open_price, close_price) - random.uniform(0, 1)
        
        history.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": random.randint(1000000, 10000000),
            "change": round((close_price - open_price) / open_price * 100, 2)
        })
    
    # 生成因子数据
    factors = {
        "rsi": round(random.uniform(30, 70), 1),
        "macd": round(random.uniform(-1, 1), 3),
        "ma5": round(base_price + random.uniform(-2, 2), 2),
        "ma20": round(base_price + random.uniform(-3, 3), 2),
        "volume_ratio": round(random.uniform(0.8, 1.2), 2),
        "pe_ratio": round(random.uniform(10, 30), 1)
    }
    
    # 生成分析报告
    report = f"""
## {symbol} 股票分析报告

### 基本信息
- **股票代码**: {symbol}
- **当前价格**: ¥{base_price:.2f}
- **涨跌幅**: {change:+.2f}%
- **AI预测**: {'看涨' if prediction == 'UP' else '看跌'}
- **模型置信度**: {confidence:.1f}%

### 技术面分析
- **RSI指标**: {factors['rsi']} ({'中性' if 30 <= factors['rsi'] <= 70 else '超买' if factors['rsi'] > 70 else '超卖'})
- **MACD**: {factors['macd']} ({'金叉' if factors['macd'] > 0 else '死叉'})
- **5日均线**: ¥{factors['ma5']}
- **20日均线**: ¥{factors['ma20']}

### 投资建议
基于AI模型分析，建议{'买入' if prediction == 'UP' else '卖出或观望'}。
本分析仅供参考，投资有风险，入市需谨慎。
"""
    
    return StockAnalysis(
        symbol=symbol,
        name=f"股票{symbol}",
        current_price=round(base_price, 2),
        price_change=round(change, 2),
        prediction=prediction,
        confidence=round(confidence, 1),
        factors=factors,
        history=history,
        report=report
    )

@app.get("/api/stock/history/{symbol}")
async def get_stock_history(
    symbol: str,
    days: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # 模拟历史数据
    import random
    from datetime import datetime, timedelta
    
    history = []
    now = datetime.now()
    base_price = random.uniform(50, 200)
    
    for i in range(days - 1, -1, -1):
        date = now - timedelta(days=i)
        open_price = base_price + random.uniform(-3, 3)
        close_price = open_price + random.uniform(-2, 2)
        high_price = max(open_price, close_price) + random.uniform(0, 1)
        low_price = min(open_price, close_price) - random.uniform(0, 1)
        
        history.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": random.randint(1000000, 10000000)
        })
    
    return {"symbol": symbol, "history": history}

# 健康检查
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# 挂载前端文件
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("启动StockAnalysis AI简化服务器...")
    print(f"前端目录: {frontend_dir}")
    print("访问地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
