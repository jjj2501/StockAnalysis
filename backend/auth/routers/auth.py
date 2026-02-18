from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import logging

from backend.auth.database import get_db, get_redis_async
from backend.auth.security import security_service
from backend.auth.schemas import (
    UserCreate, UserResponse, Token, LoginRequest, 
    ChangePassword, ForgotPasswordRequest, ResetPasswordRequest
)
from backend.auth.models import User, UserSession, AuditLog
from backend.auth.dependencies import get_current_user, get_current_user_optional
from backend.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = logging.getLogger(__name__)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """用户注册"""
    # 检查邮箱是否已存在
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已被注册"
        )
    
    # 检查用户名是否已存在
    if user_data.username:
        existing_username = db.query(User).filter(User.username == user_data.username).first()
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已被使用"
            )
    
    # 创建用户
    hashed_password = security_service.hash_password(user_data.password)
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        role=user_data.role,
        preferences={}
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # 记录审计日志
    if background_tasks:
        background_tasks.add_task(
            create_audit_log,
            db=db,
            user_id=user.id,
            action="user_registered",
            resource_type="user",
            resource_id=str(user.id),
            details={"email": user.email, "role": user.role.value}
        )
    
    logger.info(f"新用户注册: {user.email}")
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """用户登录"""
    email = form_data.username  # OAuth2使用username字段传邮箱
    password = form_data.password
    
    # 检查登录失败次数
    redis_client = await get_redis_async()
    security_service.redis = redis_client
    failures = await security_service.get_login_failures(email)
    if failures >= 5:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="登录失败次数过多，请15分钟后再试"
        )
    
    # 查找用户
    user = db.query(User).filter(User.email == email).first()
    if not user:
        await security_service.record_login_failure(email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 验证密码
    if not security_service.verify_password(password, user.hashed_password):
        await security_service.record_login_failure(email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 检查用户状态
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户已被禁用"
        )
    
    # 清除登录失败记录
    await security_service.clear_login_failures(email)
    
    # 更新最后登录时间
    user.last_login = datetime.utcnow()
    db.commit()
    
    # 创建令牌
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value
    }
    
    access_token = security_service.create_access_token(token_data)
    refresh_token = security_service.create_refresh_token(token_data)
    
    # 创建会话记录
    session = UserSession(
        user_id=user.id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    db.add(session)
    db.commit()
    
    # 记录审计日志
    if background_tasks:
        background_tasks.add_task(
            create_audit_log,
            db=db,
            user_id=user.id,
            action="user_login",
            resource_type="user",
            resource_id=str(user.id),
            details={"method": "password"}
        )
    
    logger.info(f"用户登录: {user.email}")
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """刷新访问令牌"""
    # 验证刷新令牌
    token_data = security_service.verify_refresh_token(refresh_token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌"
        )
    
    # 查找用户
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已被禁用"
        )
    
    # 检查会话是否存在
    session = db.query(UserSession).filter(
        UserSession.user_id == user.id,
        UserSession.refresh_token == refresh_token,
        UserSession.expires_at > datetime.utcnow()
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="会话已过期"
        )
    
    # 创建新的访问令牌
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value
    }
    
    new_access_token = security_service.create_access_token(token_data)
    
    # 更新会话
    session.access_token = new_access_token
    session.last_used_at = datetime.utcnow()
    db.commit()
    
    return Token(
        access_token=new_access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """用户登出"""
    # 获取当前令牌（从请求头）
    from fastapi import Request
    import fastapi
    
    # 这里简化处理，实际应该从请求头获取令牌
    # 在实际应用中，需要从请求上下文中获取令牌
    
    # 记录审计日志
    if background_tasks:
        background_tasks.add_task(
            create_audit_log,
            db=db,
            user_id=current_user.id,
            action="user_logout",
            resource_type="user",
            resource_id=str(current_user.id)
        )
    
    logger.info(f"用户登出: {current_user.email}")
    return {"message": "登出成功"}


@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """忘记密码（发送重置邮件）"""
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        # 为了防止邮箱枚举攻击，即使用户不存在也返回成功
        return {"message": "如果邮箱存在，重置链接已发送"}
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户已被禁用"
        )
    
    # 创建重置令牌（简化实现，实际应该发送邮件）
    reset_token = security_service.create_access_token({
        "sub": str(user.id),
        "email": user.email,
        "purpose": "password_reset"
    })
    
    # 记录审计日志
    if background_tasks:
        background_tasks.add_task(
            create_audit_log,
            db=db,
            user_id=user.id,
            action="password_reset_requested",
            resource_type="user",
            resource_id=str(user.id)
        )
    
    logger.info(f"密码重置请求: {user.email}")
    return {
        "message": "重置链接已发送",
        "reset_token": reset_token  # 实际应用中不应该返回令牌
    }


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """重置密码"""
    # 验证重置令牌
    try:
        from jose import jwt
        payload = jwt.decode(
            request.token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        if payload.get("purpose") != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的重置令牌"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的重置令牌"
            )
        
        # 查找用户
        from uuid import UUID
        user = db.query(User).filter(User.id == UUID(user_id)).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户已被禁用"
            )
        
        # 更新密码
        user.hashed_password = security_service.hash_password(request.new_password)
        db.commit()
        
        # 记录审计日志
        if background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="password_reset_completed",
                resource_type="user",
                resource_id=str(user.id)
            )
        
        logger.info(f"密码重置完成: {user.email}")
        return {"message": "密码重置成功"}
        
    except Exception as e:
        logger.error(f"密码重置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的重置令牌"
        )


@router.post("/change-password")
async def change_password(
    request: ChangePassword,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """修改密码"""
    # 验证当前密码
    if not security_service.verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前密码错误"
        )
    
    # 更新密码
    current_user.hashed_password = security_service.hash_password(request.new_password)
    db.commit()
    
    # 记录审计日志
    if background_tasks:
        background_tasks.add_task(
            create_audit_log,
            db=db,
            user_id=current_user.id,
            action="password_changed",
            resource_type="user",
            resource_id=str(current_user.id)
        )
    
    logger.info(f"密码修改: {current_user.email}")
    return {"message": "密码修改成功"}


# 辅助函数
def create_audit_log(
    db: Session,
    user_id: Optional[str] = None,
    action: str = "",
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None
):
    """创建审计日志"""
    try:
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details
        )
        db.add(audit_log)
        db.commit()
    except Exception as e:
        logger.error(f"创建审计日志失败: {e}")