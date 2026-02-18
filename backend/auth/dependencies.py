from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, Generator
from uuid import UUID

from backend.auth.database import get_db
from backend.auth.security import security_service
from backend.auth.schemas import TokenData, UserRole
from backend.auth.models import User
from backend.auth.database import get_redis_async

# HTTP Bearer认证方案
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取当前认证用户"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # 检查令牌是否在黑名单中
    redis_client = await get_redis_async()
    security_service.redis = redis_client
    if await security_service.is_token_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌已失效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 验证令牌
    token_data = security_service.verify_access_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 获取用户
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户已被禁用",
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """获取当前用户（可选，用于向后兼容）"""
    if credentials is None:
        return None
    
    token = credentials.credentials
    
    # 检查令牌是否在黑名单中
    redis_client = await get_redis_async()
    security_service.redis = redis_client
    if await security_service.is_token_blacklisted(token):
        return None
    
    # 验证令牌
    token_data = security_service.verify_access_token(token)
    if token_data is None:
        return None
    
    # 获取用户
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None or not user.is_active:
        return None
    
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户已被禁用")
    return current_user


def require_role(required_role: UserRole):
    """要求特定角色的装饰器"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not security_service.has_permission(current_user.role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足"
            )
        return current_user
    return role_checker


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """要求管理员权限"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user


def require_premium(current_user: User = Depends(get_current_user)) -> User:
    """要求付费用户权限"""
    if current_user.role not in [UserRole.PREMIUM, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要付费用户权限"
        )
    return current_user


# 权限检查函数
def check_permission(user: User, required_role: UserRole) -> bool:
    """检查用户是否有足够权限"""
    return security_service.has_permission(user.role, required_role)


# 审计日志依赖
async def get_audit_context(
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> dict:
    """获取审计日志上下文"""
    return {
        "user_id": current_user.id if current_user else None,
        "user_email": current_user.email if current_user else None,
        "user_role": current_user.role if current_user else None
    }