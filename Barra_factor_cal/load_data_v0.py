# 从 Mongodb 数据库中加载数据

# 导入所需的库
import pandas as pd
from pymongo import MongoClient, DESCENDING # 导入 DESCENDING 用于排序
from tqdm import tqdm # 在 Jupyter 中使用 tqdm.notebook 获得更好的进度条体验
from datetime import date
# 配置数据库连接信息

MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "barra_financial_data"


# 创建一个辅助函数，用于将集合加载到 DataFrame

def load_collection_to_df(db, collection_name: str, query: dict, projection: dict) -> pd.DataFrame:
    """加载经过筛选和投影的集合数据。"""
    print(f"正在从 '{collection_name}' 加载数据...")
    collection = db[collection_name]
    cursor = collection.find(query, projection)
    df = pd.DataFrame(list(cursor))
    print(f"-> 成功加载 {len(df):,} 行数据。")
    return df

# 连接到数据库
client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
print("成功连接到 MongoDB。")


# 定义查询时间
start_date_str = "20200101"  # 股票日线开始日期
start_date_financial_str = "20190101"  # 财务数据开始日期
end_date_str = date.today().strftime('%Y%m%d')
print(f"数据筛选范围: {start_date_str} 到 {end_date_str}")


# 选择想要查询的股票池，如：沪深300、上证50等

# 首先连接到指数成分股集合，获取成分股列表
index_components_collection_name = "index_components" #
index_code = "000300.SH"

# 1. 查找最新的交易日期
print(f"\n正在为指数 '{index_code}' 查找最新交易日期...")
collection = db[index_components_collection_name]

# 使用 find_one() 配合 sort 可以高效地找到一个文档
latest_entry = collection.find_one(
    {"index_code": index_code},      # 筛选条件
    sort=[("trade_date", DESCENDING)]       # 按 trade_date 降序排列
)

# 检查是否找到了数据
if latest_entry:
    latest_trade_date = latest_entry['trade_date']
    print(f"-> 找到最新交易日期: {latest_trade_date}")

    # 2. 使用找到的最新日期来查询所有成分股
    index_info_query = {
        "index_code": index_code,
        "trade_date": latest_trade_date  # 使用动态获取的日期
    }
    index_info_projection = {"con_code": 1, "_id": 0}

    # 调用你的辅助函数加载数据
    index_constituents_df = load_collection_to_df(
        db,
        collection_name=index_components_collection_name,
        query=index_info_query,
        projection=index_info_projection
    )

    
    constituent_list = index_constituents_df['con_code'].tolist()
    print(f"\n共获取 {len(constituent_list)} 只成分股。")

else:
    print(f"!! 未能在集合 '{index_components_collection_name}' 中找到指数 '{index_code}' 的任何记录。")

# 根据成分股列表，加载所需的股票数据
# 日线行情：close, total_mv, pb, turnover_rate, pe_ttm, circ_mv

daily_prices_query = {
    "ts_code": {"$in": constituent_list},
    "trade_date": {"$gte": start_date_str, "$lte": end_date_str}
}

daily_prices_projection = {
    "ts_code": 1,
    "trade_date": 1,
    "close": 1,
    "total_mv": 1,
    "circ_mv": 1,
    "pb": 1,
    "turnover_rate": 1,
    "pe_ttm": 1,
    "_id": 0
}

daily_prices_df = load_collection_to_df(
    db,
    collection_name="daily_prices",
    query=daily_prices_query,
    projection=daily_prices_projection
)


# 现金流量：n_cashflow_act 
cashflow_query = {
    "ts_code": {"$in": constituent_list},
    "end_date": {"$gte": start_date_financial_str, "$lte": end_date_str}
}

cashflow_projection = {
    "ts_code": 1,
    "f_ann_date": 1,
    "end_date": 1,
    "n_cashflow_act": 1,
    "_id": 0
}

cashflow_df = load_collection_to_df(
    db,
    collection_name="cashflow",
    query=cashflow_query,
    projection=cashflow_projection
)


# 财务指标： 'q_profit_yoy', 'q_sales_yoy', debt_to_assets
financial_indicators_query = {
    "ts_code": {"$in": constituent_list},
    "end_date": {"$gte": start_date_financial_str, "$lte": end_date_str}
}

financial_indicators_projection = {
    "ts_code": 1,
    "ann_date": 1, 
    "end_date": 1,
    "q_profit_yoy": 1,
    "q_sales_yoy": 1,
    "debt_to_assets": 1,
    "_id": 0
}

financial_indicators_df = load_collection_to_df(
    db,
    collection_name="financial_indicators",
    query=financial_indicators_query,
    projection=financial_indicators_projection
)




# 资产负债表： total_ncl, total_hldr_eqy_inc_min_int, 
balance_sheet_query = {
    "ts_code": {"$in": constituent_list},
    "end_date": {"$gte": start_date_financial_str, "$lte": end_date_str}
}
balance_sheet_projection = {
    "ts_code": 1,
    "f_ann_date": 1,
    "end_date": 1,
    "total_ncl": 1,
    "total_hldr_eqy_inc_min_int": 1,
    "_id": 0
}
balance_sheet_df = load_collection_to_df(
    db,
    collection_name="balancesheet",
    query=balance_sheet_query,
    projection=balance_sheet_projection 
)



# 指数数据: close

index_prices_query = {
    "ts_code": index_code,
    "trade_date": {"$gte": start_date_str, "$lte": end_date_str}            
}

index_prices_projection = {
    "ts_code": 1,
    "trade_date": 1,
    "close": 1,
    "_id": 0
}

index_prices_df = load_collection_to_df(
    db,
    collection_name="index_daily_prices",
    query=index_prices_query,
    projection=index_prices_projection
)




# 获取数据完毕后，将数据合并，涉及到时间格式

# 首先对财务数据去重
cashflow_processed_df = cashflow_df.copy()
cashflow_processed_df['f_ann_date'] = pd.to_datetime(cashflow_processed_df['f_ann_date'], format='%Y%m%d')
cashflow_processed_df['end_date'] = pd.to_datetime(cashflow_processed_df['end_date'], format='%Y%m%d')

# 先排序（按 f_ann_date 降序保留最新），然后去重保留第一条
cashflow_processed_df = cashflow_processed_df.sort_values(
    ['ts_code', 'end_date', 'f_ann_date'],
    ascending=[True, True, False]  # f_ann_date 降序，最新在前
).drop_duplicates(subset=['ts_code', 'end_date'], keep='first')

print('现金流数据去重处理与日期处理完毕')

balance_sheet_processed_df = balance_sheet_df.copy()

balance_sheet_processed_df['f_ann_date'] = pd.to_datetime(balance_sheet_processed_df['f_ann_date'], format='%Y%m%d')
balance_sheet_processed_df['end_date'] = pd.to_datetime(balance_sheet_processed_df['end_date'], format='%Y%m%d')

# 先排序（按 f_ann_date 降序保留最新），然后去重保留第一条
balance_sheet_processed_df = balance_sheet_processed_df.sort_values(
    ['ts_code', 'end_date', 'f_ann_date'],
    ascending=[True, True, False]  # f_ann_date 降序，最新在前
).drop_duplicates(subset=['ts_code', 'end_date'], keep='first')

print('资产负债表数据去重处理与日期处理完毕')

financial_indicators_processed_df = financial_indicators_df.copy()

financial_indicators_processed_df['ann_date'] = pd.to_datetime(financial_indicators_processed_df['ann_date'], format='%Y%m%d')
financial_indicators_processed_df['end_date'] = pd.to_datetime(financial_indicators_processed_df['end_date'], format='%Y%m%d')

print('财务指标数据日期处理完毕')

daily_prices_processed_df = daily_prices_df.copy()
daily_prices_processed_df['trade_date'] = pd.to_datetime(daily_prices_processed_df['trade_date'], format='%Y%m%d')

print('日线行情数据日期处理完毕')

index_prices_processed_df = index_prices_df.copy()
index_prices_processed_df['trade_date'] = pd.to_datetime(index_prices_processed_df['trade_date'], format='%Y%m%d')
print('指数行情数据日期处理完毕')


# 合并数据
'''
对于每一个 ts_code 和 trade_date，
寻找其所有财务报告中，f_ann_date（公告日）小于等于 trade_date 的所有报告，
然后在这些报告中，选择 end_date（报告期）最晚 的那一条。
'''
def robust_merge_asof(left_df, right_df, left_on, right_on, by):
    """
    A robust version of pd.merge_asof that merges group by group
    to avoid potential bugs with large DataFrames.
    """
    print(f"--- Performing robust merge_asof on '{by}' ---")
    
    # Sanitize and sort inputs
    left_df = left_df.reset_index(drop=True).sort_values(by=[by, left_on])
    right_df = right_df.reset_index(drop=True).sort_values(by=[by, right_on])
    
    all_keys = left_df[by].unique()
    merged_chunks = []
    
    for key in all_keys:
        left_chunk = left_df[left_df[by] == key]
        right_chunk = right_df[right_df[by] == key]
        
        merged_chunk = pd.merge_asof(
            left=left_chunk,
            right=right_chunk,
            left_on=left_on,
            right_on=right_on,
            by=by,
            direction='backward'
        )
        merged_chunks.append(merged_chunk)
        
    print("Robust merge successful.")
    return pd.concat(merged_chunks, ignore_index=True)

# --- How to use it ---
merged_df_1 = robust_merge_asof(
     daily_prices_processed_df,
     balance_sheet_processed_df,
     left_on='trade_date',
     right_on='f_ann_date',
     by='ts_code'
)

merged_df_1.rename(columns={
    'f_ann_date': 'balance_sheet_f_ann_date',
}, inplace=True)


merged_df_2 = robust_merge_asof(
     merged_df_1,
     financial_indicators_processed_df,
     left_on='trade_date',
     right_on='ann_date',
     by='ts_code'
)
merged_df_2.rename(columns={
    'ann_date': 'financial_indicators_ann_date',
}, inplace=True)

final_merged_df = robust_merge_asof(
     merged_df_2,
     cashflow_processed_df,
     left_on='trade_date',
     right_on='f_ann_date',
     by='ts_code'
)
final_merged_df.rename(columns={
    'f_ann_date': 'cashflow_f_ann_date',
}, inplace=True)    

stock_price_with_financials_df = final_merged_df.copy()
columns_to_drop = ['end_date_y','end_date_x'] 

stock_price_with_financials_df = stock_price_with_financials_df.drop(columns=columns_to_drop)
print(stock_price_with_financials_df.info())
print("数据合并完毕。")


# 获取完毕，数据可以直接被因子计算文件读取




