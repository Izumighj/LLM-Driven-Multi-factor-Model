import pandas as pd
import tushare as ts
from pymongo import MongoClient
from tqdm import tqdm
import time

# --- 配置 (与主脚本保持一致) ---
TUSHARE_TOKEN = '3339f390298f8503dc5cbcca9fce9898f10bd3a5cb8ce9315803e5cd'
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "barra_financial_data"

# ---【核心】定义您想回填的日期范围 ---
BACKFILL_START_DATE = '20190101'
BACKFILL_END_DATE = '20191231'

# 初始化 Tushare Pro 接口
pro = ts.pro_api(TUSHARE_TOKEN)

def backfill_historical_prices():
    """为指定历史时期回填日线行情数据"""
    
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db['daily_prices']
    print(f"Connected to MongoDB. Database: '{DB_NAME}'")

    try:
        # 1. 从 stock_info 获取需要回填的股票列表
        stock_list = list(db.stock_info.distinct("ts_code"))
        if not stock_list:
            print("错误: 'stock_info' 集合为空，无法获取股票列表。")
            return
            
        print(f"找到 {len(stock_list)} 只股票。")
        print(f"开始为所有股票回填历史数据，日期范围: {BACKFILL_START_DATE} to {BACKFILL_END_DATE}...")

        # 2. 循环为每只股票拉取指定时期的数据
        # (复用之前脚本中的智能限流和重试逻辑)
        CALLS_PER_MINUTE = 480
        call_count = 0
        start_time = time.time()
    
        for code in tqdm(stock_list, desc="Backfilling Historical Data"):
            # 检查是否需要等待
            if call_count >= CALLS_PER_MINUTE:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time + 1
                    print(f"\nRate limit reached, sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                call_count, start_time = 0, time.time()

            # API调用重试逻辑
            for attempt in range(3):
                try:
                    df = pro.daily_basic(
                        ts_code=code,
                        start_date=BACKFILL_START_DATE,
                        end_date=BACKFILL_END_DATE
                    )
                    if not df.empty:
                        records = df.to_dict('records')
                        # 使用 insert_many，唯一索引会自动处理重复数据
                        collection.insert_many(records, ordered=False)
                    
                    call_count += 1
                    time.sleep(0.125)
                    break # 成功则跳出重试
                except Exception as e:
                    # insert_many 因为重复键报错是正常的，可以忽略
                    if "duplicate key error" in str(e):
                        break 
                    print(f"\n为 {code} 回填数据时出错 (尝试 {attempt + 1}): {e}")
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"为 {code} 回填数据失败。")
                        
    except Exception as e:
        print(f"发生严重错误: {e}")
    finally:
        client.close()
        print("\n历史数据回填流程结束。")

if __name__ == '__main__':
    backfill_historical_prices()