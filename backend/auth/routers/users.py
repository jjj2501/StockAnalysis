from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import logging

from backend.auth.database import get_db
from backend.auth.schemas import UserResponse, UserUpdate, ChangePassword
from backend.auth.models import User
from backend.auth.dependencies import get_current_user, require_admin
from backend.auth.security import security_service

router = APIRouter(prefix="/users", tags=["users"])
logger = logging.getLogger(__name__)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """获取当前用户信息"""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_info(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新当前用户信息"""
    # 检查用户名是否已被使用
    if user_update.username and user_update.username != current_user.username:
        existing_user = db.query(User).filter(
            User.username == user_update.username,
            User.id != current_user.id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已被使用"
            )
    
    # 更新用户信息
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    db.commit()
    db.refresh(current_user)
    
    logger.info(f"用户信息更新: {current_user.email}")
    return current_user


@router.post("/me/change-password")
async def change_user_password(
    password_data: ChangePassword,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """修改当前用户密码"""
    # 验证当前密码
    if not security_service.verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前密码错误"
        )
    
    # 更新密码
    current_user.hashed_password = security_service.hash_password(password_data.new_password)
    db.commit()
    
    logger.info(f"用户密码修改: {current_user.email}")
    return {"message": "密码修改成功"}


@router.delete("/me")
async def delete_user_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除当前用户账户"""
    # 这里可以添加额外的确认逻辑，比如需要输入密码确认
    
    # 标记用户为禁用状态（软删除）
    current_user.is_active = False
    db.commit()
    
    logger.info(f"用户账户删除: {current_user.email}")
    return {"message": "账户已删除"}


# 管理员功能
@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """列出所有用户（仅管理员）"""
    query = db.query(User)
    
    # 应用过滤器
    if role:
        query = query.filter(User.role == role)
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    
    # 排序和分页
    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
    
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: UUID,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """获取指定用户信息（仅管理员）"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    return user


@router.put("/{user_id}/role")
async def update_user_role(
    user_id: UUID,
    role_update: dict,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """更新用户角色（仅管理员）"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 验证角色
    from backend.auth.schemas import UserRole
    new_role = role_update.get("role")
    if new_role not in [role.value for role in UserRole]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的角色"
        )
    
    # 不能修改自己的角色
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能修改自己的角色"
        )
    
    # 更新角色
    user.role = new_role
    db.commit()
    
    logger.info(f"用户角色更新: {user.email} -> {new_role}")
    return {"message": "角色更新成功", "user_id": str(user_id), "new_role": new_role}


@router.put("/{user_id}/status")
async def update_user_status(
    user_id: UUID,
    status_update: dict,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """更新用户状态（仅管理员）"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 不能修改自己的状态
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能修改自己的状态"
        )
    
    # 验证状态
    is_active = status_update.get("is_active")
    if not isinstance(is_active, bool):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的状态值"
        )
    
    # 更新状态
    user.is_active = is_active
    db.commit()
    
    action = "启用" if is_active else "禁用"
    logger.info(f"用户状态更新: {user.email} -> {action}")
    return {"message": f"用户已{action}", "user_id": str(user_id), "is_active": is_active}


@router.get("/stats/summary")
async def get_user_stats_summary(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """获取用户统计摘要（仅管理员）"""
    from sqlalchemy import func
    
    # 总用户数
    total_users = db.query(func.count(User.id)).scalar()
    
    # 活跃用户数
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    
    # 按角色统计
    role_stats = db.query(
        User.role,
        func.count(User.id).label("count")
    ).group_by(User.role).all()
    
    # 今日新增用户
    from datetime import datetime, timedelta
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    new_users_today = db.query(func.count(User.id)).filter(
        User.created_at >= today_start
    ).scalar()
    
    # 最近7天活跃用户（有登录记录）
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_active_users = db.query(func.count(User.id)).filter(
        User.last_login >= seven_days_ago
    ).scalar()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "inactive_users": total_users - active_users,
        "role_distribution": {role: count for role, count in role_stats},
        "new_users_today": new_users_today,
        "recent_active_users": recent_active_users,
        "active_rate": (recent_active_users / total_users * 100) if total_users > 0 else 0
    }