from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.core.engine import StockEngine
from backend.core.llm import get_llm_client
import os

router = APIRouter()
engine = StockEngine()

# 默认使用 ollama, 可通过环境变量配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
llm_client = get_llm_client(provider=LLM_PROVIDER)

@router.get("/predict/{symbol}")
async def predict_stock(symbol: str):
    """
    获取股票预测结果
    """
    try:
        result = engine.predict(symbol)
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
    获取股票深度分析报告 (含 LLM)
    """
    try:
        # 1. 获取基础数据和预测
        data_result = engine.predict(symbol)
        if "error" in data_result:
             raise HTTPException(status_code=400, detail=data_result["error"])
             
        # 2. 调用 LLM 生成报告
        report = llm_client.generate_report(symbol, data_result)
        
        return {
            "symbol": symbol,
            "data": data_result,
            "report": report
        }
    except Exception as e:
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
