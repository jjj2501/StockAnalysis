from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./stockanalysis.db")
    
    # Redis配置
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # JWT配置
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-here-change-in-production")
    JWT_REFRESH_SECRET_KEY: str = os.getenv("JWT_REFRESH_SECRET_KEY", "your-refresh-secret-key-here")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # 安全配置
    PASSWORD_HASH_ALGORITHM: str = "argon2"
    ARGON2_TIME_COST: int = 2
    ARGON2_MEMORY_COST: int = 102400
    ARGON2_PARALLELISM: int = 8
    
    # 应用配置
    APP_NAME: str = "StockAnalysis AI"
    API_V1_STR: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # 邮箱配置（用于密码重置）
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST")
    SMTP_PORT: Optional[int] = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
    EMAILS_FROM_EMAIL: Optional[str] = os.getenv("EMAILS_FROM_EMAIL")
    
    # 速率限制
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # GPU/训练配置
    USE_GPU: bool = os.getenv("USE_GPU", "True").lower() == "true"
    GPU_DEVICE_ID: int = int(os.getenv("GPU_DEVICE_ID", "0"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    EPOCHS: int = int(os.getenv("EPOCHS", "50"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    MODEL_HIDDEN_DIM: int = int(os.getenv("MODEL_HIDDEN_DIM", "64"))
    MODEL_NUM_LAYERS: int = int(os.getenv("MODEL_NUM_LAYERS", "2"))
    SEQUENCE_LENGTH: int = int(os.getenv("SEQUENCE_LENGTH", "60"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()