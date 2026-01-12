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
