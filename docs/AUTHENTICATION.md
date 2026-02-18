# StockAnalysis AI 认证系统文档

## 概述

StockAnalysis AI 现已集成完整的用户认证和管理系统，提供安全的用户登录、注册、会话管理和权限控制功能。

## 技术架构

### 认证流程
1. **用户注册** → 创建用户账户（密码使用Argon2哈希）
2. **用户登录** → 验证凭证，返回JWT访问令牌和刷新令牌
3. **API访问** → 使用访问令牌调用受保护API
4. **令牌刷新** → 访问令牌过期后使用刷新令牌获取新访问令牌
5. **用户登出** → 使令牌失效

### 安全特性
- **密码哈希**: Argon2算法（抗GPU/ASIC攻击）
- **令牌管理**: JWT + 刷新令牌模式
- **速率限制**: 防止暴力破解攻击
- **审计日志**: 记录所有认证相关操作
- **安全头**: 添加安全相关的HTTP头

## API 端点

### 认证端点

#### 1. 用户注册
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "username": "username"
}
```

**响应**:
```json
{
  "message": "用户注册成功",
  "user_id": 1,
  "email": "user@example.com",
  "username": "username"
}
```

#### 2. 用户登录
```http
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded

email=user@example.com&password=SecurePassword123!
```

**响应**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "username": "username",
    "role": "free"
  }
}
```

#### 3. 刷新访问令牌
```http
POST /api/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

**响应**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### 4. 用户登出
```http
POST /api/auth/logout
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

#### 5. 忘记密码
```http
POST /api/auth/forgot-password
Content-Type: application/json

{
  "email": "user@example.com"
}
```

#### 6. 重置密码
```http
POST /api/auth/reset-password
Content-Type: application/json

{
  "token": "<reset_token>",
  "new_password": "NewSecurePassword123!"
}
```

#### 7. 修改密码
```http
POST /api/auth/change-password
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "current_password": "OldPassword123!",
  "new_password": "NewSecurePassword123!"
}
```

### 用户管理端点

#### 1. 获取当前用户信息
```http
GET /api/users/me
Authorization: Bearer <access_token>
```

**响应**:
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "username",
  "role": "free",
  "created_at": "2024-01-01T00:00:00",
  "last_login": "2024-01-01T12:00:00"
}
```

#### 2. 获取所有用户（管理员）
```http
GET /api/users/
Authorization: Bearer <access_token>
```

#### 3. 更新用户信息
```http
PUT /api/users/{user_id}
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "username": "new_username",
  "role": "premium"
}
```

## 数据库模型

### User 表
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    role VARCHAR(20) DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

### UserSession 表
```sql
CREATE TABLE user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### AuditLog 表
```sql
CREATE TABLE audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action VARCHAR(100) NOT NULL,
    details TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 配置

### 环境变量
```bash
# 数据库配置
DATABASE_URL=sqlite:///./stockanalysis.db

# Redis配置（用于令牌黑名单和缓存）
REDIS_URL=redis://localhost:6379/0

# JWT配置
JWT_SECRET_KEY=your-secret-key-here-change-in-production
JWT_REFRESH_SECRET_KEY=your-refresh-secret-key-here

# 应用配置
DEBUG=false

# 邮箱配置（用于密码重置）
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAILS_FROM_EMAIL=your-email@gmail.com
```

### 配置文件
所有配置在 `backend/config.py` 中定义，支持环境变量覆盖。

## 前端集成

### 认证状态管理
前端使用localStorage存储令牌和用户信息：
```javascript
// 存储认证信息
localStorage.setItem('access_token', accessToken);
localStorage.setItem('refresh_token', refreshToken);
localStorage.setItem('username', username);
localStorage.setItem('user_id', userId);

// 检查认证状态
function checkAuthStatus() {
    const accessToken = localStorage.getItem('access_token');
    const refreshToken = localStorage.getItem('refresh_token');
    // ... 更新UI显示
}

// 登出
function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('username');
    localStorage.removeItem('user_id');
    // ... 更新UI并重定向
}
```

### 受保护API调用
```javascript
async function callProtectedAPI() {
    const accessToken = localStorage.getItem('access_token');
    
    const response = await fetch('/api/protected-endpoint', {
        headers: {
            'Authorization': `Bearer ${accessToken}`
        }
    });
    
    if (response.status === 401) {
        // 令牌过期，尝试刷新
        await refreshToken();
        return callProtectedAPI(); // 重试
    }
    
    return response.json();
}
```

## 测试

### 单元测试
```bash
# 运行认证系统测试
python scripts/test_auth.py

# 运行完整认证流程测试
python scripts/test_complete_auth.py
```

### 手动测试
1. 启动服务器：`uvicorn backend.main:app --reload`
2. 访问 `http://localhost:8000/docs` 查看API文档
3. 使用前端页面测试完整流程

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查 `DATABASE_URL` 配置
   - 确保数据库文件可写

2. **Redis连接失败**
   - 检查Redis服务是否运行
   - 验证 `REDIS_URL` 配置

3. **JWT令牌无效**
   - 检查令牌是否过期
   - 验证 `JWT_SECRET_KEY` 配置

4. **密码重置邮件不发送**
   - 检查SMTP配置
   - 验证邮箱服务凭据

### 日志查看
认证相关日志记录在：
- 控制台输出（DEBUG模式）
- 数据库audit_logs表
- 应用日志文件

## 安全最佳实践

1. **生产环境**：
   - 修改默认JWT密钥
   - 使用强密码策略
   - 启用HTTPS
   - 配置适当的速率限制

2. **密码安全**：
   - 使用Argon2哈希算法
   - 强制密码复杂度
   - 定期要求密码更改

3. **会话管理**：
   - 使用短寿命访问令牌（15分钟）
   - 使用长寿命刷新令牌（7天）
   - 实现令牌黑名单

4. **审计与监控**：
   - 记录所有认证事件
   - 监控异常登录尝试
   - 定期审查审计日志