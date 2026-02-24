import akshare as ak
try:
    df = ak.stock_a_indicator_lg(symbol="600519")
    print(df.tail(1).to_dict('records'))
except Exception as e:
    print("Error:", e)
