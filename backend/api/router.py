from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from backend.core.engine import StockEngine
from backend.core.llm import get_llm_client
from backend.core.progress import progress_manager
import os
import pandas as pd
from typing import Optional

router = APIRouter()
engine = StockEngine()

# LLM 配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")
llm_client = get_llm_client(provider=LLM_PROVIDER, model_name=LLM_MODEL)

@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """
    SSE 端点：获取任务实时进度
    """
    return StreamingResponse(progress_manager.subscribe(task_id), media_type="text/event-stream")

@router.get("/predict/{symbol}")
async def predict_stock(symbol: str):
    """
    获取股票预测结果 (支持可选 task_id 用于追踪进度)
    """
    try:
        # 这里预测是异步的，需要等待
        result = await engine.predict(symbol, task_id=symbol)
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{symbol}")
async def analyze_stock(symbol: str):
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
        
        return {
            "symbol": symbol,
            "data": data_result,
            "report": report
        }
    except Exception as e:
        await progress_manager.update(symbol, 0, "error", str(e))
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{symbol}")
async def get_history(symbol: str, days: int = 30):
    """
    获取股票历史 K 线数据
    """
    try:
        import datetime
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days + 60)).strftime("%Y%m%d") # 多取一些确保指标计算
        
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
                
        return {
            "symbol": symbol,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        print(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/factors/{symbol}")
async def get_factors(symbol: str, cat: Optional[str] = None):
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
        return result
    except Exception as e:
        print(f"Factors fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest/{symbol}")
async def backtest_stock(
    symbol: str, 
    start_date: str = "20230101", 
    end_date: str = "20240101", 
    strategy_type: str = "macd",
    initial_cash: float = 100000.0
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
async def analyze_backtest_results(request: BacktestAnalysisRequest):
    """
    基于回测结果生成 AI 诊断报告 (按需触发)
    """
    try:
        report = llm_client.generate_backtest_report(
            request.symbol, 
            request.strategy, 
            request.summary
        )
        return {"report": report}
    except Exception as e:
        print(f"Backtest analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
