import pandas as pd
import tushare as ts
from pymongo import MongoClient
from tqdm import tqdm
import time
from datetime import date, timedelta

# --- 配置 (与主脚本保持一致) ---
TUSHARE_TOKEN = '3339f390298f8503dc5cbcca9fce9898f10bd3a5cb8ce9315803e5cd'
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "barra_financial_data"

# 初始化 Tushare Pro 接口
pro = ts.pro_api(TUSHARE_TOKEN)

def fill_missing_daily_prices():
    """查找并补充 daily_prices 集合中缺失的股票数据"""
    
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    print(f"Connected to MongoDB. Database: '{DB_NAME}'")

    try:
        # 1. 获取应该有的全量股票列表
        all_stocks_cursor = db.stock_info.find({}, {"ts_code": 1, "_id": 0})
        full_stock_list = {item['ts_code'] for item in all_stocks_cursor}
        if not full_stock_list:
            print("错误: 'stock_info' 集合为空或不存在，无法进行比较。")
            return
        print(f"应有股票总数: {len(full_stock_list)}")

        # 2. 获取已经存在的股票列表
        existing_stock_list = set(db.daily_prices.distinct("ts_code"))
        print(f"已有数据的股票数量: {len(existing_stock_list)}")

        # 3. 计算出缺失的股票列表
        missing_stocks = list(full_stock_list - existing_stock_list)
        print(f"缺失数据的股票数量: {len(missing_stocks)}")

        if missing_stocks:
            print("\n--- 以下是缺失的具体股票代码 ---")
            print(missing_stocks)
            print("---------------------------------")
        
        if not missing_stocks:
            print("数据完整，无需补充！")
            return

        # 4. 只为缺失的股票拉取数据
        # 注意：这里我们假设是从头开始拉取，所以start_date是固定的
        start_date = '20200101'
        end_date = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
        print(f"开始为 {len(missing_stocks)} 只缺失的股票补充数据，日期范围: {start_date} to {end_date}...")
        
        # (这里可以复用您主脚本中的智能限流和重试逻辑)
        for code in tqdm(missing_stocks, desc="Filling Missing Data"):
            try:
                df = pro.daily_basic(ts_code=code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    records = df.to_dict('records')
                    db.daily_prices.insert_many(records, ordered=False)
                #time.sleep(0.2) # 遵守流量限制
            except Exception as e:
                print(f"为 {code} 补充数据时出错: {e}")

    finally:
        client.close()
        print("\n查缺补漏流程结束。")

if __name__ == '__main__':
    fill_missing_daily_prices()