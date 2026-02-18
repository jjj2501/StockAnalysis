from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Generator
import redis.asyncio as redis
from backend.config import settings

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis连接池
redis_pool = None


def get_redis() -> redis.Redis:
    """获取Redis连接"""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    return redis_pool


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_redis_async() -> redis.Redis:
    """获取异步Redis连接"""
    return get_redis()


def init_db():
    """初始化数据库，创建所有表"""
    from backend.auth.models import Base
    Base.metadata.create_all(bind=engine)
    print("数据库表创建完成")


def drop_db():
    """删除所有表（仅用于测试）"""
    from backend.auth.models import Base
    Base.metadata.drop_all(bind=engine)
    print("数据库表已删除")