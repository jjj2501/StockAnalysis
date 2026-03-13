from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
import time
from pathlib import Path
import pandas as pd
import io
from typing import List

from backend.core.data import DataFetcher, DEFAULT_CACHE_DIR

router = APIRouter()
fetcher = DataFetcher()

class FetchRequest(BaseModel):
    symbol: str
    market: str

class FetchMacroRequest(BaseModel):
    market: str

@router.get("/cache")
async def list_cache():
    """获取所有本地缓存的数据文件列表（涵盖K线、因子JSON与宏观JSON）"""
    if not DEFAULT_CACHE_DIR.exists():
        return []
    
    files_info = []
    
    # 扫描外层 K线 parquet
    for file_path in DEFAULT_CACHE_DIR.glob("*.parquet"):
        try:
            stat = file_path.stat()
            size_kb = stat.st_size / 1024
            modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            
            filename = file_path.name
            market = "CN"
            symbol = filename.replace(".parquet", "")
            if "_" in symbol:
                parts = symbol.split("_", 1)
                market = parts[0]
                symbol = parts[1]

            files_info.append({
                "type": "kline",
                "filename": filename,
                "symbol": symbol,
                "market": market,
                "size_kb": round(size_kb, 2),
                "modified_time": modified_time
            })
        except Exception:
            pass

    # 扫描 factors json
    factors_dir = DEFAULT_CACHE_DIR / "factors"
    if factors_dir.exists():
        for file_path in factors_dir.glob("*.json"):
            try:
                stat = file_path.stat()
                size_kb = stat.st_size / 1024
                modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                
                filename = f"factors/{file_path.name}"
                market = "CN"
                symbol = file_path.name.replace(".json", "")
                if "_" in symbol:
                    parts = symbol.split("_", 1)
                    market = parts[0]
                    symbol = parts[1]

                files_info.append({
                    "type": "factor",
                    "filename": filename,
                    "symbol": symbol,
                    "market": market,
                    "size_kb": round(size_kb, 2),
                    "modified_time": modified_time
                })
            except Exception:
                pass

    # 扫描 macro json
    macro_dir = DEFAULT_CACHE_DIR / "macro"
    if macro_dir.exists():
        for file_path in macro_dir.glob("*.json"):
            try:
                stat = file_path.stat()
                size_kb = stat.st_size / 1024
                modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                
                filename = f"macro/{file_path.name}"
                market = file_path.name.replace("macro_", "").replace(".json", "")

                files_info.append({
                    "type": "macro",
                    "filename": filename,
                    "symbol": "MACRO-ALL",
                    "market": market,
                    "size_kb": round(size_kb, 2),
                    "modified_time": modified_time
                })
            except Exception:
                pass

    files_info.sort(key=lambda x: x["modified_time"], reverse=True)
    return files_info

@router.delete("/cache/{filename:path}")
async def delete_cache(filename: str):
    """删除指定的缓存文件 (支持相对路径如 factors/xxx.json)"""
    # 此处防范跨目录漏洞
    safe_filename = filename.replace("..", "").strip("/")
    file_path = DEFAULT_CACHE_DIR / safe_filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Cache file not found")
    
    try:
        os.remove(file_path)
        return {"status": "success", "message": f"Deleted {safe_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@router.post("/fetch")
async def force_fetch_data(req: FetchRequest):
    """强制更新某资产的数据并覆写缓存"""
    start_date = "2020-01-01"
    end_date = time.strftime("%Y-%m-%d") # 今天
    
    try:
        # 直接调用底层的从远程获取的方法，跳过缓存读取
        df = fetcher._fetch_from_remote(req.symbol, start_date, end_date, req.market)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data returned from upstream for {req.symbol} ({req.market})")
        
        # 强制保存覆盖
        success = fetcher._save_cache(req.symbol, df, req.market)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save cache file")
            
        return {
            "status": "success",
            "message": f"Updated data for {req.symbol}",
            "rows": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_csv_data(
    symbol: str = Form(...), 
    market: str = Form(...),
    file: UploadFile = File(...)
):
    """手动上传 CSV 清洗好的文件来强行覆盖本地 Parquet 黑盒"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # 简单清洗列名以防万一
        df.columns = [c.lower().strip() for c in df.columns]
        
        required_cols = {"date", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing}")
            
        # 规范化日期格式
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values("date")
        
        # 保存覆盖
        success = fetcher._save_cache(symbol, df, market)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save uploaded data to parquet cache")
            
        return {
            "status": "success",
            "message": f"Successfully overlaid data for {symbol}",
            "rows": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@router.post("/fetch_macro")
async def force_fetch_macro(req: FetchMacroRequest):
    """强制更新某大盘的宏观数据字典并覆写缓存"""
    try:
        # 为了强制从外部更新，我们先删除本地当天的宏观文件命中可能
        import os
        macro_path = fetcher.cache_dir / "macro" / f"macro_{req.market}.json"
        if macro_path.exists():
            os.remove(macro_path)
            
        macro_data = fetcher.get_macro_data(req.market)
        if not macro_data:
             raise HTTPException(status_code=400, detail=f"Failed to fetch upstream macro data for {req.market}")
        return {
            "status": "success",
            "message": f"Updated globally macro data for {req.market}",
            "keys": list(macro_data.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch_factors")
async def force_fetch_factors(req: FetchRequest):
    """强制更新某资产的全维度量化因子数据并覆写每日JSON缓存"""
    try:
        import os
        factor_path = fetcher._get_factor_cache_path(req.symbol, req.market)
        if factor_path.exists():
            os.remove(factor_path)
            
        factors = fetcher.get_comprehensive_factors(f"{req.market}_{req.symbol}" if req.market!="CN" else req.symbol)
        
        if "error" in factors and len(factors.keys()) <= 1:
            raise HTTPException(status_code=500, detail=f"Factor update failed: {factors.get('error')}")
            
        return {
            "status": "success",
            "message": f"Updated comprehensive factors for {req.symbol}",
            "keys": list(factors.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
