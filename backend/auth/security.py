from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID
import redis.asyncio as redis
from backend.config import settings
from backend.auth.schemas import TokenData, UserRole
import logging

logger = logging.getLogger(__name__)

# 密码哈希上下文
pwd_context = CryptContext(
    schemes=["argon2"],
    argon2__time_cost=settings.ARGON2_TIME_COST,
    argon2__memory_cost=settings.ARGON2_MEMORY_COST,
    argon2__parallelism=settings.ARGON2_PARALLELISM,
    deprecated="auto"
)


class SecurityService:
    """安全服务类"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
    
    # 密码相关方法
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)
    
    # JWT相关方法
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.JWT_SECRET_KEY, 
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """创建刷新令牌"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.JWT_REFRESH_SECRET_KEY, 
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt
    
    def verify_access_token(self, token: str) -> Optional[TokenData]:
        """验证访问令牌"""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            if payload.get("type") != "access":
                return None
            
            user_id = payload.get("sub")
            if user_id is None:
                return None
            
            return TokenData(
                user_id=UUID(user_id),
                email=payload.get("email"),
                role=payload.get("role")
            )
        except JWTError as e:
            logger.error(f"JWT验证失败: {e}")
            return None
    
    def verify_refresh_token(self, token: str) -> Optional[TokenData]:
        """验证刷新令牌"""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_REFRESH_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            if payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            if user_id is None:
                return None
            
            return TokenData(
                user_id=UUID(user_id),
                email=payload.get("email"),
                role=payload.get("role")
            )
        except JWTError as e:
            logger.error(f"刷新令牌验证失败: {e}")
            return None
    
    # Redis黑名单管理
    async def add_to_blacklist(self, token: str, expire_minutes: int = 15) -> bool:
        """将令牌加入黑名单"""
        if not self.redis:
            return False
        
        try:
            # 计算剩余过期时间
            payload = jwt.get_unverified_claims(token)
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                exp_time = datetime.fromtimestamp(exp_timestamp)
                now = datetime.utcnow()
                if exp_time > now:
                    ttl = int((exp_time - now).total_seconds())
                    await self.redis.setex(f"blacklist:{token}", ttl, "1")
                    return True
        except Exception as e:
            logger.error(f"加入黑名单失败: {e}")
        
        return False
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """检查令牌是否在黑名单中"""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.get(f"blacklist:{token}")
            return result is not None
        except Exception as e:
            logger.error(f"检查黑名单失败: {e}")
            return False
    
    # 登录保护
    async def record_login_failure(self, email: str) -> int:
        """记录登录失败"""
        if not self.redis:
            return 0
        
        key = f"login_failures:{email}"
        try:
            failures = await self.redis.incr(key)
            # 设置15分钟过期
            await self.redis.expire(key, 900)
            return failures
        except Exception as e:
            logger.error(f"记录登录失败失败: {e}")
            return 0
    
    async def get_login_failures(self, email: str) -> int:
        """获取登录失败次数"""
        if not self.redis:
            return 0
        
        key = f"login_failures:{email}"
        try:
            failures = await self.redis.get(key)
            return int(failures) if failures else 0
        except Exception as e:
            logger.error(f"获取登录失败次数失败: {e}")
            return 0
    
    async def clear_login_failures(self, email: str) -> bool:
        """清除登录失败记录"""
        if not self.redis:
            return False
        
        key = f"login_failures:{email}"
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"清除登录失败记录失败: {e}")
            return False
    
    # 权限检查
    def has_permission(self, user_role: UserRole, required_role: UserRole) -> bool:
        """检查用户是否有足够权限"""
        role_hierarchy = {
            UserRole.FREE: 0,
            UserRole.PREMIUM: 1,
            UserRole.ADMIN: 2
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level


# 全局安全服务实例
security_service = SecurityService()