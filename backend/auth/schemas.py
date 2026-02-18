from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


class UserRole(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ADMIN = "admin"


class UserBase(BaseModel):
    email: EmailStr
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    is_active: Optional[bool] = True
    is_verified: Optional[bool] = False
    role: Optional[UserRole] = UserRole.FREE


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码至少需要8个字符')
        if not any(c.isupper() for c in v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含至少一个数字')
        return v


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    preferences: Optional[Dict[str, Any]] = None


class UserInDB(UserBase):
    id: UUID
    created_at: datetime
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True


class UserResponse(UserInDB):
    pass


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    user_id: Optional[UUID] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChangePassword(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('新密码至少需要8个字符')
        if not any(c.isupper() for c in v):
            raise ValueError('新密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('新密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('新密码必须包含至少一个数字')
        return v


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('新密码至少需要8个字符')
        if not any(c.isupper() for c in v):
            raise ValueError('新密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('新密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('新密码必须包含至少一个数字')
        return v


class UserRoleUpdate(BaseModel):
    role: UserRole


class AuditLogCreate(BaseModel):
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class AuditLogResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True