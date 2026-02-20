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

# LLM 配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")
llm_client = get_llm_client(provider=LLM_PROVIDER, model_name=LLM_MODEL)

logger = logging.getLogger(__name__)


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
        report = llm_client.generate_report(symbol, data_result)
        
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
        df = fetcher.get_stock_data(symbol, start_date=start_date, end_date=end_date)
        
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
        fetcher = DataFetcher()
        
        categories = cat.split(',') if cat else None
        result = fetcher.get_comprehensive_factors(symbol, categories=categories)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
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
        # 增加一些前置数据用于计算指标
        import datetime
        dt_start = datetime.datetime.strptime(start_date, "%Y%m%d")
        dt_pre = (dt_start - datetime.timedelta(days=60)).strftime("%Y%m%d")
        
        df = fetcher.get_stock_data(symbol, start_date=dt_pre, end_date=end_date)
        if df.empty:
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
        report = llm_client.generate_backtest_report(
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
    shares: int
    price: float

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
        
        # 2. 调用 LLM 生成分析报告
        report = llm_client.generate_portfolio_risk_report(portfolio_data, risk_result)
        
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