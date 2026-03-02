from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
from typing import Dict, Any, Optional
import json

from backend.auth.database import get_redis_async
from backend.auth.security import security_service

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.public_paths = {
            "/api/auth/",
            "/api/health",
            "/",
            "/index.html",
            "/login.html",
            "/register.html",
            "/style.css",
            "/app.js"
        }
    
    async def dispatch(self, request: Request, call_next):
        # 记录请求开始时间
        start_time = time.time()
        
        # 检查是否为公开路径
        path = request.url.path
        is_public = any(path.startswith(public_path) for public_path in self.public_paths)
        
        # 如果不是公开路径，检查认证
        if not is_public and path.startswith("/api/"):
            auth_header = request.headers.get("Authorization")
            
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]  # 去掉"Bearer "前缀
                
                # 检查令牌是否在黑名单中
                redis_client = await get_redis_async()
                security_service.redis = redis_client
                
                if await security_service.is_token_blacklisted(token):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "令牌已失效"}
                    )
                
                # 验证令牌
                token_data = security_service.verify_access_token(token)
                if token_data is None:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "无效的认证令牌"}
                    )
                
                # 将用户信息添加到请求状态
                request.state.user_id = token_data.user_id
                request.state.user_email = token_data.email
                request.state.user_role = token_data.role
        
        # 处理请求
        try:
            response = await call_next(request)
        except HTTPException as e:
            response = JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            logger.error(f"请求处理失败: {e}", exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={"detail": "内部服务器错误"}
            )
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录访问日志
        self.log_access(request, response, process_time)
        
        return response
    
    def log_access(self, request: Request, response, process_time: float):
        """记录访问日志"""
        try:
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": round(process_time, 3),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
                "user_id": getattr(request.state, "user_id", None),
                "user_email": getattr(request.state, "user_email", None),
                "user_role": getattr(request.state, "user_role", None)
            }
            
            # 根据状态码选择日志级别
            if response.status_code >= 500:
                logger.error(f"访问日志: {json.dumps(log_data)}")
            elif response.status_code >= 400:
                logger.warning(f"访问日志: {json.dumps(log_data)}")
            else:
                logger.info(f"访问日志: {json.dumps(log_data)}")
                
        except Exception as e:
            logger.error(f"记录访问日志失败: {e}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.public_paths = {
            "/api/auth/",
            "/api/health",
            "/"
        }
    
    async def dispatch(self, request: Request, call_next):
        # 公开路径不限制
        path = request.url.path
        is_public = any(path.startswith(public_path) for public_path in self.public_paths)
        
        if is_public:
            return await call_next(request)
        
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"
        
        # 获取用户ID（如果有）
        user_id = getattr(request.state, "user_id", None)
        
        # 使用用户ID或IP作为限制键
        limit_key = f"user:{user_id}" if user_id else f"ip:{client_ip}"
        
        # 检查速率限制
        redis_client = await get_redis_async()
        
        try:
            # 使用滑动窗口算法
            current_time = int(time.time())
            window_key = f"ratelimit:{limit_key}:{current_time // 60}"
            
            # 增加当前分钟的计数
            current_count = await redis_client.incr(window_key)
            
            # 设置过期时间（2分钟，确保跨分钟边界）
            await redis_client.expire(window_key, 120)
            
            # 检查是否超过限制
            if current_count > self.requests_per_minute:
                # 获取用户信息用于日志
                user_email = getattr(request.state, "user_email", "anonymous")
                
                logger.warning(
                    f"速率限制触发: {limit_key}, "
                    f"用户: {user_email}, "
                    f"路径: {path}, "
                    f"计数: {current_count}"
                )
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "请求过于频繁，请稍后再试",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
        except Exception as e:
            logger.error(f"速率限制检查失败: {e}")
            # 如果Redis失败，暂时跳过速率限制
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全头中间件"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # 添加安全相关的HTTP头
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:;",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """审计中间件"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_paths = [
            "/api/",
        ]
        self.exclude_paths = [
            "/api/auth/login",
            "/api/auth/register",
            "/api/health"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # 检查是否需要审计
        path = request.url.path
        should_audit = any(path.startswith(audit_path) for audit_path in self.audit_paths)
        should_exclude = any(path == exclude_path for exclude_path in self.exclude_paths)
        
        if should_audit and not should_exclude:
            # 收集审计信息
            audit_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": request.method,
                "path": path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
                "user_id": getattr(request.state, "user_id", None),
                "user_email": getattr(request.state, "user_email", None),
                "user_role": getattr(request.state, "user_role", None)
            }
            
            # 对于POST/PUT/PATCH请求，记录请求体（敏感信息需要过滤）
            # 临时禁用中间件中的 request.body() 读取，避免引发 Starlette 请求流死锁挂起问题
            # if request.method in ["POST", "PUT", "PATCH"]:
            #     try:
            #         body = await request.body()
            #         if body:
            #             # 过滤敏感信息
            #             body_str = body.decode()
            #             filtered_body = self.filter_sensitive_data(body_str)
            #             audit_data["request_body"] = filtered_body
            #     except Exception:
            #         pass
            
            # 记录审计日志
            logger.info(f"审计日志: {json.dumps(audit_data)}")
        
        return await call_next(request)
    
    def filter_sensitive_data(self, body_str: str) -> str:
        """过滤敏感信息"""
        try:
            import json as json_module
            data = json_module.loads(body_str)
            
            # 过滤密码字段
            if isinstance(data, dict):
                for key in ["password", "current_password", "new_password", "confirm_password"]:
                    if key in data:
                        data[key] = "***FILTERED***"
                
                return json_module.dumps(data)
        except:
            pass
        
        return body_str