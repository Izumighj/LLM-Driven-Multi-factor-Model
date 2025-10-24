import pandas as pd
import tushare as ts
from pymongo import MongoClient, DESCENDING
from tqdm import tqdm
import time
from datetime import date, timedelta
import tushare_fetcher as fetcher

# --- 配置 ---
TUSHARE_TOKEN = '3339f390298f8503dc5cbcca9fce9898f10bd3a5cb8ce9315803e5cd'

# --- MongoDB 配置 ---
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/" # 本地MongoDB连接字符串
DB_NAME = "barra_financial_data"                      # 数据库名称

# 初始化 Tushare Pro 接口
pro = ts.pro_api(TUSHARE_TOKEN)

def get_last_update_date(db, collection_name, date_col='trade_date'):
    """从MongoDB集合中获取某个字段的最新日期"""
    collection = db[collection_name]
    try:
        # 寻找按日期降序排列的第一个文档
        latest_record = collection.find_one(sort=[(date_col, DESCENDING)])
        if latest_record and date_col in latest_record:
            return pd.to_datetime(latest_record[date_col])
    except Exception as e:
        print(f"Error reading last date from {collection_name}: {e}")
    # 如果集合不存在、为空或出错，则返回一个初始日期
    return pd.to_datetime('20190101')

def update_stock_info(db):
    """
    获取最新的所有A股列表并存入MongoDB。
    每次都用最新的列表覆盖旧集合。
    """
    COLLECTION_NAME = 'stock_info'
    collection = db[COLLECTION_NAME]
    print(f"--- Starting update for {COLLECTION_NAME} ---")
    try:
        stock_df = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        
        if not stock_df.empty:
            # 1. 删除旧的集合
            collection.drop()
            # 2. 将DataFrame转换为字典列表
            records = stock_df.to_dict('records')
            # 3. 插入新数据
            collection.insert_many(records)
            print(f"Successfully saved {len(stock_df)} stocks to '{COLLECTION_NAME}' collection.")
            return stock_df['ts_code'].tolist()
        else:
            print("Warning: Fetched stock list is empty.")
            return []
    except Exception as e:
        print(f"An error occurred while updating stock info: {e}")
        return []

def update_daily_prices(db):
    """
    Optimized function to update daily prices for all stocks.
    It fetches data day-by-day for all stocks at once, reducing API calls.
    """
    COLLECTION_NAME = 'daily_prices'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting optimized update for {COLLECTION_NAME} ---")

    # For preventing duplicate data, a unique index on (ts_code, trade_date) is recommended.
    # You only need to run this once in the MongoDB shell:
    # db.daily_prices.createIndex({ ts_code: 1, trade_date: 1 }, { unique: true })

    # 1. Get the last update date from the database
    last_date = get_last_update_date(db, COLLECTION_NAME, 'trade_date')
    if last_date == pd.to_datetime('20190101'):
        start_date = last_date.strftime('%Y%m%d')
    else:
        start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
    
    end_date = date.today().strftime('%Y%m%d')

    if start_date > end_date:
        print("Daily prices are already up to date.")
        return

    # 2. Fetch the list of all trading days in the required date range
    try:
        trade_cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        trade_dates = trade_cal_df[trade_cal_df['is_open'] == 1]['cal_date'].tolist()
        if not trade_dates:
            print(f"No trading days found between {start_date} and {end_date}.")
            return
    except Exception as e:
        print(f"Error fetching trade calendar: {e}")
        return
        
    print(f"Fetching data for {len(trade_dates)} trading days from {start_date} to {end_date}...")

    # 3. Loop through each trading day and fetch data for all stocks
    all_data = []
    for trade_day in tqdm(trade_dates, desc="Updating Daily Prices by Date"):
        try:
            df = pro.daily_basic(trade_date=trade_day, fields=[
                "ts_code", "trade_date", "close", "turnover_rate",
                "turnover_rate_f", "volume_ratio", "pe", "pe_ttm",
                "pb", "ps", "ps_ttm", "dv_ratio",
                "dv_ttm", "total_share", "float_share", "free_share",
                "total_mv", "circ_mv"
            ])
            if not df.empty:
                all_data.append(df)
            
            # Tushare's daily_basic has a rate limit (e.g., 5000 records per call).
            # A small delay is still good practice to be safe.
            time.sleep(0.2) 
        except Exception as e:
            print(f"Error fetching daily prices for trade_date {trade_day}: {e}")

    # 4. Concatenate and insert data into MongoDB
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        if not final_df.empty:
            records = final_df.to_dict('records')
            try:
                # 'ordered=False' allows the operation to continue even if some records fail (e.g., duplicates)
                collection.insert_many(records, ordered=False)
                print(f"Successfully inserted/updated {len(records)} rows into {COLLECTION_NAME}.")
            except Exception as e:
                print(f"An error occurred during bulk insert: {e}. Some duplicates may have been skipped due to unique index.")
        else:
            print("No new daily price data was fetched.")
    else:
        print("No new daily price data was fetched.")

def update_financial_indicators(db, stock_list):
    """
    通过遍历股票列表，更新所有股票的财务指标数据。
    """
    if not stock_list:
        print("Stock list is empty, skipping financial indicators update.")
        return

    COLLECTION_NAME = 'financial_indicators'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} ---")

    # 强烈建议为 (ts_code, end_date) 创建唯一索引以防重复
    # 只需在MongoDB中执行一次:
    # db.financial_indicators.createIndex({ ts_code: 1, end_date: 1 }, { unique: true })

    # 复用智能限流和重试逻辑
    CALLS_PER_MINUTE = 480
    call_count = 0
    start_time = time.time()

    for code in tqdm(stock_list, desc="Updating Financial Indicators"):
        if call_count >= CALLS_PER_MINUTE:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time + 1
                print(f"\nRate limit reached, sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            call_count, start_time = 0, time.time()
        
        for attempt in range(3):
            try:
                # 调用 tushare_fetcher 中更新后的、包含完整字段的函数
                df = fetcher.fetch_financial_indicators_by_stock(code)
                
                if not df.empty:
                    records = df.to_dict('records')
                    collection.insert_many(records, ordered=False)
                
                call_count += 1
                time.sleep(0.125)
                break
            except Exception as e:
                if "duplicate key error" in str(e).lower():
                    break
                
                print(f"\nError fetching financial data for {code} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"Failed to fetch data for {code} after 3 attempts.")

def update_balancesheet(db, stock_list):
    """
    通过遍历股票列表，更新所有股票的资产负债表数据。
    """
    if not stock_list:
        print("Stock list is empty, skipping balance sheet update.")
        return

    COLLECTION_NAME = 'balancesheet'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} ---")

    # 强烈建议为 (ts_code, end_date) 创建唯一索引以防重复
    # 只需在MongoDB中执行一次:
    # db.balancesheet.createIndex({ ts_code: 1, end_date: 1 }, { unique: true })

    # Reuse the smart rate-limiting and retry logic
    CALLS_PER_MINUTE = 480
    call_count = 0
    start_time = time.time()

    for code in tqdm(stock_list, desc="Updating Balance Sheets"):
        # Check if we need to wait
        if call_count >= CALLS_PER_MINUTE:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time + 1
                print(f"\nRate limit reached, sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            call_count, start_time = 0, time.time()
        
        # API call retry logic
        for attempt in range(3):
            try:
                # Call the new fetcher function
                df = fetcher.fetch_balancesheet_by_stock(code)
                
                if not df.empty:
                    records = df.to_dict('records')
                    # The unique index will automatically handle duplicates
                    collection.insert_many(records, ordered=False)
                
                call_count += 1
                time.sleep(0.125) # Basic delay
                break # Success, break the retry loop
            except Exception as e:
                # Ignore duplicate key errors, which are expected when data already exists
                if "duplicate key error" in str(e).lower():
                    break
                
                print(f"\nError fetching balance sheet for {code} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(5) # Wait 5 seconds before retrying
                else:
                    print(f"Failed to fetch balance sheet for {code} after 3 attempts.")


def update_cashflow(db, stock_list):
    """
    通过遍历股票列表，更新所有股票的现金流量表数据。
    """
    if not stock_list:
        print("Stock list is empty, skipping cash flow statement update.")
        return

    COLLECTION_NAME = 'cashflow'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} ---")

    # It's highly recommended to create a unique index to prevent duplicates
    # Execute this once in MongoDB Shell:
    # db.cashflow.createIndex({ ts_code: 1, end_date: 1 }, { unique: true })

    # Reuse the smart rate-limiting and retry logic
    CALLS_PER_MINUTE = 480
    call_count = 0
    start_time = time.time()

    for code in tqdm(stock_list, desc="Updating Cash Flow Statements"):
        # Check if we need to wait
        if call_count >= CALLS_PER_MINUTE:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time + 1
                print(f"\nRate limit reached, sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            call_count, start_time = 0, time.time()
        
        # API call retry logic
        for attempt in range(3):
            try:
                # Call the new fetcher function
                df = fetcher.fetch_cashflow_by_stock(code)
                
                if not df.empty:
                    records = df.to_dict('records')
                    # The unique index will automatically handle duplicates
                    collection.insert_many(records, ordered=False)
                
                call_count += 1
                time.sleep(0.125) # Basic delay
                break # Success, break the retry loop
            except Exception as e:
                # Ignore duplicate key errors, which are expected when data already exists
                if "duplicate key error" in str(e).lower():
                    break
                
                print(f"\nError fetching cash flow statement for {code} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(5) # Wait 5 seconds before retrying
                else:
                    print(f"Failed to fetch cash flow statement for {code} after 3 attempts.")

def update_income(db, stock_list):
    """
     通过遍历股票列表，更新所有股票的利润表数据。
    """
    if not stock_list:
        print("Stock list is empty, skipping income statement update.")
        return

    COLLECTION_NAME = 'income'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} ---")

    # It's highly recommended to create a unique index to prevent duplicates
    # Execute this once in MongoDB Shell:
    # db.income.createIndex({ ts_code: 1, end_date: 1 }, { unique: true })

    # Reuse the smart rate-limiting and retry logic
    CALLS_PER_MINUTE = 480
    call_count = 0
    start_time = time.time()

    for code in tqdm(stock_list, desc="Updating Income Statements"):
        # Check if we need to wait
        if call_count >= CALLS_PER_MINUTE:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time + 1
                print(f"\nRate limit reached, sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            call_count, start_time = 0, time.time()
        
        # API call retry logic
        for attempt in range(3):
            try:
                # Call the new fetcher function
                df = fetcher.fetch_income_by_stock(code)
                
                if not df.empty:
                    records = df.to_dict('records')
                    # The unique index will automatically handle duplicates
                    collection.insert_many(records, ordered=False)
                
                call_count += 1
                time.sleep(0.125) # Basic delay
                break # Success, break the retry loop
            except Exception as e:
                # Ignore duplicate key errors, which are expected when data already exists
                if "duplicate key error" in str(e).lower():
                    break
                
                print(f"\nError fetching income statement for {code} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(5) # Wait 5 seconds before retrying
                else:
                    print(f"Failed to fetch income statement for {code} after 3 attempts.")



# 指数基本信息更新函数
def update_index_info(db):
    """
    获取最新的所有指数基本信息并存入MongoDB。
    每次都用最新的列表覆盖旧集合。
    """
    COLLECTION_NAME = 'index_info'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} ---")
    try:
        # 1. 调用 fetcher 获取数据
        index_df = fetcher.fetch_index_info()
        
        if not index_df.empty:
            # 2. 删除旧的集合
            collection.drop()
            # 3. 将DataFrame转换为字典列表
            records = index_df.to_dict('records')
            # 4. 插入新数据
            collection.insert_many(records)
            print(f"Successfully saved {len(index_df)} indices to '{COLLECTION_NAME}' collection.")
        else:
            print("Warning: Fetched index list is empty.")
    except Exception as e:
        print(f"An error occurred while updating index info: {e}")

# 指数日行情数据更新函数

# 指定几个指数：沪深300、上证50、中证100

def update_daily_index_prices(db, index_list):
    """更新对应的日线行情数据到MongoDB（包含智能限流与重试）"""
    if not index_list:
        print("Index list is empty, skipping daily prices update.")
        return

    COLLECTION_NAME = 'index_daily_prices'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} ---")
    
    last_date = get_last_update_date(db, COLLECTION_NAME, 'trade_date')
    start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
    end_date = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
    
    if start_date > end_date:
        print("Index daily prices are already up to date.")
        return

    print(f"Fetching data from {start_date} to {end_date}...")
    
    all_data = []
    
    # --- 【核心优化】智能限流逻辑 ---
    # Tushare pro 一般每分钟限制200次
    CALLS_PER_MINUTE = 190  # 我们设置得保守一些，比如190次，留出安全边际
    call_count = 0
    start_time = time.time()
    
    for code in tqdm(index_list, desc="Updating Daily Index Prices"):
        # 1. 检查是否达到一分钟内的调用上限
        if call_count >= CALLS_PER_MINUTE:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < 60:
                # 如果用时不到60秒，就暂停剩余的时间
                sleep_time = 60 - elapsed_time + 1 # 多等1秒确保安全
                print(f"\nRate limit reached for the minute, sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            # 2. 重置计数器和计时器，开始新的时间窗口
            call_count = 0
            start_time = time.time()

        # --- 【新增】API调用重试逻辑 ---
        for attempt in range(3): # 最多重试3次
            try:
                df = fetcher.fetch_daily_index_prices(code, start_date, end_date)
                all_data.append(df)
                call_count += 1 # 3. API调用成功，计数器+1
                break # 成功则跳出重试循环
            except Exception as e:
                print(f"\nError fetching index daily prices for {code} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    print("Retrying after 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to fetch data for {code} after 3 attempts.")
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        if not final_df.empty:
            records = final_df.to_dict('records')
            try:
                collection.insert_many(records, ordered=False)
                print(f"\nSuccessfully inserted/updated {len(records)} rows into {COLLECTION_NAME}.")
            except Exception as e:
                print(f"An error occurred during bulk insert: {e}. Some duplicates may have been skipped.")
        else:
            print("No new index daily price data was fetched.")


# 指数成分股数据更新函数

def update_index_components(db, index_list):
    """更新指定指数的成分股数据，并实现智能限流与重试"""
    if not index_list:
        print("Index list is empty, skipping index components update.")
        return

    COLLECTION_NAME = 'index_components'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} ---")

    # --- 【核心优化】智能限流与重试逻辑 ---
    # Tushare pro 一般每分钟限制200次
    CALLS_PER_MINUTE = 190  # 我们设置得保守一些，比如190次，留出安全边际
    call_count = 0
    start_time = time.time()
    
    all_data = []

    for index_code in tqdm(index_list, desc="Updating Index Components"):
        # 1. 检查是否达到一分钟内的调用上限
        if call_count >= CALLS_PER_MINUTE:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < 60:
                # 如果用时不到60秒，就暂停剩余的时间
                sleep_time = 60 - elapsed_time + 1 # 多等1秒确保安全
                print(f"\nRate limit reached for the minute, sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            # 2. 重置计数器和计时器，开始新的时间窗口
            call_count = 0
            start_time = time.time()

        # --- 【新增】API调用重试逻辑 ---
        for attempt in range(3): # 最多重试3次
            try:
                df = fetcher.fetch_index_components(index_code)
                if not df.empty:
                    all_data.append(df)
                
                call_count += 1 # 3. API调用成功，计数器+1
                break # 成功则跳出重试循环
            except Exception as e:
                print(f"\nError fetching components for index {index_code} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    print("Retrying after 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to fetch data for index {index_code} after 3 attempts.")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        if not final_df.empty:
            # 获取那些成功抓取到数据的指数代码
            updated_indices = final_df['index_code'].unique().tolist()
            
            # 仅删除已成功更新的指数的旧成分股数据
            if updated_indices:
                print(f"Deleting old components for {len(updated_indices)} indices...")
                collection.delete_many({"index_code": {"$in": updated_indices}})
                
                records = final_df.to_dict('records')
                collection.insert_many(records)
                print(f"Successfully updated/inserted components for {len(updated_indices)} indices.")
            else:
                print("No new index component data was formatted for insertion.")
        else:
            print("No index component data was fetched.")
    else:
        print("No index component data was fetched.")


# 申万行业成分股数据
# update_mongo_db.py

# ... (文件其他部分代码不变) ...

# 申万行业成分股数据 (从CSV文件导入的新版本)
def update_sw_industries_from_csv(db, csv_file_path: str):
    """
    从本地的CSV文件加载申万行业分类数据并存入MongoDB。
    每次都用CSV文件的最新数据覆盖旧集合。

    参数:
    db: MongoDB数据库连接对象。
    csv_file_path (str): CSV文件的完整路径。
    """
    COLLECTION_NAME = 'sw_industries'
    collection = db[COLLECTION_NAME]
    print(f"\n--- Starting update for {COLLECTION_NAME} from CSV file: {csv_file_path} ---")

    try:
        # 1. 使用 pandas 读取CSV文件
        sw_df = pd.read_csv(csv_file_path)
        
        if not sw_df.empty:
            # 2. 删除旧的集合以确保数据从零开始
            print(f"Dropping old collection '{COLLECTION_NAME}'...")
            collection.drop()
            
            # 3. 将DataFrame转换为字典列表
            records = sw_df.to_dict('records')
            
            # 4. 插入新数据
            collection.insert_many(records)
            print(f"Successfully saved {len(records)} SW industry classification records to '{COLLECTION_NAME}'.")
        else:
            print("Warning: The CSV file is empty. No data was inserted.")
            
    except FileNotFoundError:
        print(f"ERROR: The file was not found at the specified path: {csv_file_path}")
    except Exception as e:
        print(f"An error occurred while updating SW industries from CSV: {e}")

# ... (文件其他部分代码不变) ...

# 中信行业成分股数据

# --- 主函数 ---
if __name__ == '__main__':
    # 创建MongoDB客户端连接
    client = MongoClient(MONGO_CONNECTION_STRING)
    # 选择数据库
    db = client[DB_NAME]
    print(f"Connected to MongoDB. Database: '{DB_NAME}'")
    
    # 步骤 1: 更新股票基本信息集合，并获取股票列表
    stock_list_to_update = update_stock_info(db)
    
    # 步骤 2: 使用获取到的股票列表，更新日线行情数据
    update_daily_prices(db)

    # 步骤 3: 更新财务指标数据
    #update_financial_indicators(db, stock_list_to_update) 

    # 步骤 4: 更新资产负债表数据
    #update_balancesheet(db, stock_list_to_update)

    # 步骤 5: 更新现金流量表数据
    #update_cashflow(db, stock_list_to_update)

    # 步骤 6: 更新利润表数据
    #update_income(db, stock_list_to_update)

    # 步骤 7: 更新指数基本信息
    #update_index_info(db)

    # 步骤 8: 更新指定指数的日线行情数据
    index_list = ['000300.SH','000016.SH','000903.SH'] #沪深300、上证50、中证100
    update_daily_index_prices(db, index_list)

    # 步骤 9: 更新指定指数的成分股数据
    #update_index_components(db, index_list)

    # 步骤 10: 更新申万行业分类数据
    #sw_industry_csv_path = 'Barra_factor_cal/data/stk_sw_industry.csv' 
    #update_sw_industries_from_csv(db, sw_industry_csv_path)
    
    # 关闭数据库连接
    client.close()
    print("\nDatabase update process finished.")