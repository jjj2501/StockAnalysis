from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import os
import pandas as pd
import logging

from backend.core.engine import StockEngine
from backend.core.llm import get_llm_client
from backend.core.progress import progress_manager
from backend.auth.dependencies import get_current_user_optional
from backend.auth.models import User
from backend.auth.database import get_db
from backend.auth.schemas import AuditLogCreate

router = APIRouter()
engine = StockEngine()

# LLM 动态配置
# 后续所有接口均应在请求时通过 get_llm_client 动态获取并加载 settings 变量，不再使用全局对象

logger = logging.getLogger(__name__)


from pydantic import BaseModel

class LLMConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None

@router.get("/config/llm")
async def get_llm_config():
    """获取当前大模型的外部配置"""
    from backend.config import settings
    return {
        "api_key": settings.OPENAI_API_KEY or "",
        "base_url": settings.OPENAI_BASE_URL or ""
    }

@router.post("/config/llm")
async def update_llm_config(config: LLMConfig):
    """更新大模型 API 密钥与网关并在后台文件持久化"""
    from backend.config import settings
    
    # 存入缓存文件并刷新当前运行时全局参数
    settings.save_llm_cache(
        api_key=config.api_key or "",
        base_url=config.base_url or ""
    )
    return {"message": "大模型网关设定已生效并持久化"}

class LLMTestConfig(BaseModel):
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

@router.post("/config/llm/test")
async def test_llm_connection(config: LLMTestConfig):
    """测试给定的大模型配置连通性"""
    try:
        from backend.core.llm import get_llm_client
        client = get_llm_client(
            provider=config.provider, 
            model_name=config.model_name, 
            api_key=config.api_key, 
            base_url=config.base_url
        )
        
        # 针对大模型发送极简握手
        messages = [{"role": "user", "content": "Hello! Please reply with exactly one word: 'OK'."}]
        response = client.chat_with_tools(messages=messages)
        
        if response and response.get("content"):
            reply = response.get('content', '').strip()
            # 由于可能出现错误捕获被放成 fallback [外部大脑短路：...]
            if "[外部大脑短路" in reply or "[内部通信总线崩塌" in reply:
                return {"status": "error", "message": f"连接发生异常或超时: {reply}"}
            return {"status": "success", "message": f"连接成功！(回复: {reply})"}
        else:
            return {"status": "error", "message": "连接成功但未获得有效响应内容"}
    except Exception as e:
        return {"status": "error", "message": f"连接失败: {str(e)}"}

@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """
    SSE 端点：获取任务实时进度
    """
    return StreamingResponse(progress_manager.subscribe(task_id), media_type="text/event-stream")


@router.get("/predict/{symbol}")
async def predict_stock(
    symbol: str,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    获取股票预测结果 (支持可选 task_id 用于追踪进度)
    """
    try:
        # 这里预测是异步的，需要等待
        result = await engine.predict(symbol, task_id=symbol)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 记录用户活动（如果有用户）
        if user and background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="stock_prediction",
                resource_type="stock",
                resource_id=symbol,
                details={
                    "symbol": symbol,
                    "prediction": result.get("predicted_trend"),
                    "confidence": result.get("confidence")
                }
            )
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    获取股票深度分析报告 (含进度上报)
    """
    try:
        # 1. 进度：开始 (10%)
        # 2. 获取预测结果并上报进度
        data_result = await engine.predict(symbol, task_id=symbol)
        if "error" in data_result:
            raise HTTPException(status_code=400, detail=data_result["error"])
        
        await progress_manager.update(symbol, 95, "正在通过人工智能生成深度分析报告...")
        
        # 3. 调用 LLM 生成报告
        from backend.config import settings
        from backend.core.llm import get_llm_client
        act_provider = getattr(settings, "LLM_PROVIDER", "ollama")
        act_model = getattr(settings, "LLM_MODEL", "qwen3:1.7b")
        if act_provider == "ollama" and getattr(settings, "OPENAI_API_KEY", None):
             pass
        
        client = get_llm_client(provider=act_provider, model_name=act_model)
        report = client.generate_report(symbol, data_result)
        
        await progress_manager.update(symbol, 100, "分析完成")
        
        # 记录用户活动（如果有用户）
        if user and background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="stock_analysis",
                resource_type="stock",
                resource_id=symbol,
                details={
                    "symbol": symbol,
                    "analysis_type": "full",
                    "has_llm_report": True
                }
            )
        
        return {
            "symbol": symbol,
            "data": data_result,
            "report": report
        }
    except Exception as e:
        await progress_manager.update(symbol, 0, "error", str(e))
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_history(
    symbol: str,
    days: int = 30,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    获取股票历史 K 线数据
    """
    try:
        import datetime
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days + 60)).strftime("%Y%m%d")  # 多取一些确保指标计算
        
        from backend.core.data import DataFetcher
        fetcher = DataFetcher()
        clean_symbol, market = fetcher.parse_symbol_market(symbol)
        df = fetcher.get_stock_data(clean_symbol, start_date=start_date, end_date=end_date, market=market)
        
        if df.empty:
            return {"symbol": symbol, "history": []}
        
        # 只取最近的 days 天
        df = df.tail(days)
        
        # 转换为列表字典格式供前端使用
        history = df.to_dict(orient='records')
        # 处理 Timestamp 转 string
        for item in history:
            if 'date' in item and hasattr(item['date'], 'strftime'):
                item['date'] = item['date'].strftime('%Y-%m-%d')
        
        # 记录用户活动（如果有用户）
        if user and background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="view_stock_history",
                resource_type="stock",
                resource_id=symbol,
                details={
                    "symbol": symbol,
                    "days": days,
                    "record_count": len(history)
                }
            )
        
        return {
            "symbol": symbol,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/factors/{symbol}")
async def get_factors(
    symbol: str,
    cat: Optional[str] = None,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    获取综合量化因子 (支持分类过滤)
    :param cat: 逗号分隔的类别, 如 "tech,fundamental"
    """
    try:
        from backend.core.data import DataFetcher
        import math
        fetcher = DataFetcher()
        
        categories = cat.split(',') if cat else None
        result = fetcher.get_comprehensive_factors(symbol, categories=categories)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 清理 NaN/Inf 值，防止 JSON 序列化失败
        def sanitize(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return round(obj, 4)
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(item) for item in obj]
            return obj
        result = sanitize(result)
        
        # 记录用户活动（如果有用户）
        if user and background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="view_stock_factors",
                resource_type="stock",
                resource_id=symbol,
                details={
                    "symbol": symbol,
                    "categories": categories,
                    "factor_count": len(result.get("factors", []))
                }
            )
        
        return result
    except Exception as e:
        logger.error(f"Factors fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/factors/{symbol}/analyze")
async def analyze_factors_stream(
    symbol: str,
    cat: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
):
    """
    量化因子大模型流式诊断
    接收 symbol 和分类信息，获取最新因子雷达图，交由 LLM 进行打字机式(SSE)诊断分析推流
    """
    try:
        from backend.core.data import DataFetcher
        fetcher = DataFetcher()
        categories = cat.split(',') if cat else None
        
        # 拉取因子数据
        factors_data = fetcher.get_comprehensive_factors(symbol, categories=categories)
        if "error" in factors_data:
            def error_streamer():
                yield f"data: ⚠️ 无法获取因子数据: {factors_data['error']}\n\n"
            return StreamingResponse(error_streamer(), media_type="text/event-stream")
            
        import json
        from backend.core.llm import get_llm_client
        from backend.config import settings
        
        # 智能 fallback 回退逻辑
        act_provider = provider if provider and provider != "null" else getattr(settings, "LLM_PROVIDER", "ollama")
        act_model = model if model and model != "null" else getattr(settings, "LLM_MODEL", "qwen3:1.7b")
        # 特别兼容：若配置了 OPENAI_API_KEY 但没设置 default provider 为 openai 的容错
        if act_provider == "ollama" and getattr(settings, "OPENAI_API_KEY", None):
             # 用户明显填了 key 却没改默认，自动帮他切过去 (防止一直由于连不上 ollama 而崩溃)
             pass # 为了不那么魔改，这里暂不强制截获，只做普通回退
             
        def streamer():
            client = get_llm_client(provider=act_provider, model_name=act_model)
            for chunk in client.generate_factor_report_stream(symbol, factors_data):
                # 遵循 SSE 标准格式，用 JSON 包装能够安全传递换行等特殊符号，防范前端错误累加空行
                payload = json.dumps({"text": str(chunk)})
                yield f"data: {payload}\n\n"
        
        return StreamingResponse(streamer(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Factors analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{symbol}/stream")
async def multi_agent_safari_stream(symbol: str, provider: Optional[str] = None, model: Optional[str] = None):
    """
    多智能体 (Multi-Agent) 联合诊断流式接口。
    负责拉拔数据并在流中吐出宏观分析师、量化研究员等角色的对话式弹珠。
    """
    try:
        from backend.core.agents.orch import AgentOrchestrator
        from backend.config import settings
        
        act_provider = provider if provider and provider != "null" else getattr(settings, "LLM_PROVIDER", "ollama")
        act_model = model if model and model != "null" else getattr(settings, "LLM_MODEL", "qwen3:1.7b")
        
        orch = AgentOrchestrator(provider=act_provider, model_name=act_model)
        # FastAPI 会自动将同步 Generator 交由线程池执行，不会阻塞主事件循环
        return StreamingResponse(orch.run_safari(symbol), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Multi-Agent Safari error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/memory/{symbol}")
async def get_agent_memory(symbol: str):
    """
    获取指定股票的历史推演记忆摘要（供前端历史洞察面板展示）
    """
    try:
        from backend.core.agents.memory import AgentMemoryStore
        history = AgentMemoryStore.load_history(symbol, top_k=5)
        symbols = AgentMemoryStore.list_all_symbols()
        return {"symbol": symbol, "history": history, "all_symbols": symbols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/insights")
async def get_global_insights():
    """
    获取跨股票全局智慧库（所有推演中积累的通用规律洞见）
    """
    try:
        from backend.core.agents.memory import AgentMemoryStore
        insights = AgentMemoryStore.get_global_insights(top_k=20)
        return {"insights": insights, "count": len(insights)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/backtest/{symbol}")
async def backtest_stock(
    symbol: str,
    start_date: str = "20230101",
    end_date: str = "20240101",
    strategy_type: str = "macd",
    initial_cash: float = 100000.0,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    股票策略回测接口
    """
    try:
        from backend.core.data import DataFetcher
        from backend.core.backtester import BacktestEngine, MACDStrategy, RSIStrategy, AIStrategy
        
        fetcher = DataFetcher()
        # 智能解析清洗符号后缀与市场 (解决美股/港股识别问题)
        clean_symbol, market = fetcher.parse_symbol_market(symbol)
        
        # 增加一些前置数据用于计算指标
        import datetime
        dt_start = datetime.datetime.strptime(start_date, "%Y%m%d")
        dt_pre = (dt_start - datetime.timedelta(days=60)).strftime("%Y%m%d")
        
        df = fetcher.get_stock_data(clean_symbol, start_date=dt_pre, end_date=end_date, market=market)
        if df.empty:
            logger.error(f"Backtest engine failed to retrieve data for {symbol} ({clean_symbol}/{market}) from {dt_pre} to {end_date}")
            raise HTTPException(status_code=400, detail="No data found for backtest")
        
        df = fetcher.add_technical_indicators(df)
        
        # 截取用户要求的实际回测区间
        df = df[df['date'] >= pd.to_datetime(start_date)]
        
        # 选择策略
        if strategy_type == "macd":
            strategy = MACDStrategy()
        elif strategy_type == "rsi":
            strategy = RSIStrategy()
        elif strategy_type == "ai":
            strategy = AIStrategy(engine, symbol)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported strategy: {strategy_type}")
        
        backtester = BacktestEngine(initial_cash=initial_cash)
        result = backtester.run(df, strategy)
        
        # 记录用户活动（如果有用户）
        if user and background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="run_backtest",
                resource_type="stock",
                resource_id=symbol,
                details={
                    "symbol": symbol,
                    "strategy": strategy_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "initial_cash": initial_cash,
                    "performance": result.get("summary", {})
                }
            )
        
        return {
            "symbol": symbol,
            "strategy": strategy_type,
            "params": {"start": start_date, "end": end_date, "cash": initial_cash},
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel


class BacktestAnalysisRequest(BaseModel):
    symbol: str
    strategy: str
    summary: dict
    provider: Optional[str] = None
    model: Optional[str] = None


@router.post("/backtest/analyze")
async def analyze_backtest_results(
    request: BacktestAnalysisRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    基于回测结果生成 AI 诊断报告 (按需触发)
    """
    try:
        from backend.core.llm import get_llm_client
        from backend.config import settings
        
        act_provider = request.provider if request.provider and request.provider != "null" else getattr(settings, "LLM_PROVIDER", "ollama")
        act_model = request.model if request.model and request.model != "null" else getattr(settings, "LLM_MODEL", "qwen3:1.7b")
        
        client = get_llm_client(provider=act_provider, model_name=act_model)
        
        report = client.generate_backtest_report(
            request.symbol,
            request.strategy,
            request.summary
        )
        
        # 记录用户活动（如果有用户）
        if user and background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="analyze_backtest",
                resource_type="stock",
                resource_id=request.symbol,
                details={
                    "symbol": request.symbol,
                    "strategy": request.strategy,
                    "has_llm_report": True
                }
            )
        
        return {"report": report}
    except Exception as e:
        logger.error(f"Backtest analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 用户专属功能
@router.get("/user/portfolio")
async def get_user_portfolio(
    user: User = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    获取用户的投资组合（需要认证）
    """
    if not user:
        raise HTTPException(
            status_code=401,
            detail="需要登录才能访问投资组合"
        )
    
    # 这里可以添加获取用户投资组合的逻辑
    # 暂时返回空列表
    return {
        "user_id": str(user.id),
        "portfolios": [],
        "total_value": 0
    }

from typing import List

class PortfolioItem(BaseModel):
    symbol: str
    shares: float
    price: float
    market: str = "CN"
    asset_type: str = "STOCK"
    currency: str = "CNY"

class PortfolioRiskRequest(BaseModel):
    portfolio: List[PortfolioItem]
    force_refresh: bool = False

# 简单的内存缓存
# 结构: {"hash_key_YYYYMMDD": risk_result_dict}
_RISK_CACHE = {}

import hashlib
import json
import datetime

def _get_portfolio_cache_key(portfolio_data: List[dict]) -> str:
    # 按照 symbol 排序保证顺序无关
    sorted_p = sorted(portfolio_data, key=lambda x: x["symbol"])
    json_str = json.dumps(sorted_p)
    # 附加上当天的日期作为缓存失效机制（每天必然要重新算一次最新的）
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    md5_hash = hashlib.md5(json_str.encode('utf-8')).hexdigest()
    return f"{md5_hash}_{today_str}"

@router.post("/portfolio/risk")
async def calculate_portfolio_risk(
    request: PortfolioRiskRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    计算投资组合风控指标 (VaR, CVaR) 并生成 AI 风险诊断报告 (带本地缓存)
    """
    try:
        portfolio_data = [item.dict() for item in request.portfolio]
        if not portfolio_data:
            raise HTTPException(status_code=400, detail="投资组合为空")
            
        cache_key = _get_portfolio_cache_key(portfolio_data)
        
        # 1. 尝试读缓存
        if not request.force_refresh and cache_key in _RISK_CACHE:
            logger.info("命中投资组合风控缓存，直接返回")
            return _RISK_CACHE[cache_key]
            
        logger.info("未命中缓存或要求强制刷新，开始后台计算投资组合风控并请求 LLM...")
        
        from backend.core.risk import RiskManager
        risk_manager = RiskManager()
        
        # 2. 计算各个维度的风控数据
        risk_result = risk_manager.calculate_portfolio_risk(portfolio_data)

        if "error" in risk_result:
            raise HTTPException(status_code=400, detail=risk_result["error"])
        
        from backend.core.llm import get_llm_client
        from backend.config import settings
        act_provider = getattr(settings, "LLM_PROVIDER", "ollama")
        act_model = getattr(settings, "LLM_MODEL", "qwen3:1.7b")
        if act_provider == "ollama" and getattr(settings, "OPENAI_API_KEY", None):
             pass
        
        # 2. 调用 LLM 生成分析报告
        client = get_llm_client(provider=act_provider, model_name=act_model)
        report = client.generate_portfolio_risk_report(portfolio_data, risk_result)
        
        # 将生成的报告存入结果中
        risk_result["ai_report"] = report
        risk_result["cached"] = False # 标记本次是否是从缓存读的
        
        # 更新缓存
        _RISK_CACHE[cache_key] = risk_result
        
        # 拷贝一份避免后续被改
        response_data = dict(risk_result)
        response_data["cached"] = False
        if not request.force_refresh and cache_key in _RISK_CACHE:
            _RISK_CACHE[cache_key]["cached"] = True
        
        # 记录用户活动
        if user and background_tasks:
            background_tasks.add_task(
                create_audit_log,
                db=db,
                user_id=user.id,
                action="portfolio_risk_analysis",
                resource_type="portfolio",
                details={
                    "item_count": len(portfolio_data),
                    "total_value": risk_result.get("total_value"),
                    "var_99": risk_result.get("metrics", {}).get("historical", {}).get("var_99")
                }
            )
            
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio risk calculation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/history")
async def get_user_history(
    skip: int = 0,
    limit: int = 50,
    user: User = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    获取用户的分析历史（需要认证）
    """
    if not user:
        raise HTTPException(
            status_code=401,
            detail="需要登录才能访问历史记录"
        )
    
    # 这里可以添加获取用户历史记录的逻辑
    # 暂时返回空列表
    return {
        "user_id": str(user.id),
        "total_count": 0,
        "history": []
    }


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
        from backend.auth.models import AuditLog
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