# main.py

import pandas as pd
import os
import config  # 导入配置文件
from factor_calculator import FactorCalculator
from post_processing import winsorize_factors, calculate_composite_factors, orthogonalize_factors
from load_data import load_and_prepare_data
from pymongo import MongoClient
import gc

MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "barra_financial_data"


def save_df_to_mongodb(db, df: pd.DataFrame, collection_name: str):
    """
    将DataFrame的数据完整地保存到指定的MongoDB集合中。
    此操作会先删除旧的集合，再插入新数据。
    """
    if df.empty:
        print(f"Warning: DataFrame for collection '{collection_name}' is empty. Nothing to save.")
        return

    collection = db[collection_name]
    print(f"\n--- Saving data to MongoDB collection: '{collection_name}' ---")

    try:
        # 1. 删除旧集合，确保数据完全更新
        collection.drop()
        
        # 2. 将DataFrame转换为字典列表
        records = df.to_dict('records')
        
        # 3. 插入新数据
        collection.insert_many(records)
        print(f"Successfully saved {len(records)} records to '{collection_name}'.")
        
    except Exception as e:
        print(f"An error occurred while saving to '{collection_name}': {e}")

def main():
    """主流程控制函数"""
    
    # 1. 准备数据
    print("Step 1: Loading and preparing data from MongoDB...")
    stk_data, index_data, sw_industry_data = load_and_prepare_data()

    # 2. 计算原始因子
    print("\nStep 2: Calculating raw factors...")
    calculator = FactorCalculator(prices_df=stk_data, index_df=index_data)

    # <--- 2. 内存清理 ---
    # stk_data 已经被 calculator 复制，可以删除
    del stk_data
    del index_data
    gc.collect()
    # --- 结束修改 ---

    raw_factors_df = calculator.run(config.FACTORS_TO_RUN)
    # <--- 3. 内存清理 ---
    del calculator  # calculator 对象不再需要，释放
    gc.collect()
    # --- 结束修改 ---
    print(raw_factors_df.info())

    # 3. 因子后处理流程
    print("\nStep 3: Post-processing factors...")
    
    # 3.1 去极值化
    factor_columns = [col for col in raw_factors_df.columns if col not in ['ts_code', 'trade_date']]
    winsorized_df = winsorize_factors(raw_factors_df, factor_columns)
    # <--- 4. 内存清理 ---
    del raw_factors_df
    gc.collect()
    # --- 结束修改 ---
    
    # 3.2 因子合成
    composite_df = calculate_composite_factors(winsorized_df, config.COMPOSITE_CONFIG)
    # <--- 5. 内存清理 ---
    del winsorized_df
    gc.collect()
    # --- 结束修改 ---

    # 3.3 因子正交化
    processed_df = orthogonalize_factors(composite_df, config.ORTHO_RULES)
    # <--- 6. 内存清理 ---
    del composite_df
    gc.collect()
    # --- 结束修改 ---

    # 4. 准备Barra格式输出
    print("\nStep 4: Preparing final Barra format...")
    
    # 4.1 合并收益率、市值和行业信息
    #price_info_df = calculator.prices_df[['ts_code', 'trade_date', 'ret', 'circ_mv']]
    #barra_df = pd.merge(processed_df, price_info_df, on=['ts_code', 'trade_date'], how='left')
    barra_df = pd.merge(processed_df, sw_industry_data[['ts_code', 'l1_code']], on='ts_code', how='left')
    barra_df['ret'] = barra_df.groupby('ts_code')['ret'].shift(-1) # 获取下一期收益率
    # <--- 8. 内存清理 ---
    del processed_df
   
    gc.collect()
    # --- 结束修改 ---
    
    # 4.2 重命名列
    barra_df.rename(columns=config.COLUMN_RENAME_MAP, inplace=True)
    
    # 4.3 按指定顺序选择最终列
    # <--- 9. 优化: 检查列是否存在 ---
    final_columns = [col for col in config.BARRA_OUTPUT_COLUMNS if col in barra_df.columns]
    final_barra_df = barra_df[final_columns]
    # --- 结束修改 ---

    # 5. 保存结果
    print("\nStep 5: Saving results...")
    os.makedirs(config.RESULT_DIR, exist_ok=True) # 确保结果文件夹存在
    
    
    # 5.1 保存Barra因子数据
    #barra_data_path = os.path.join(config.RESULT_DIR, 'barra_factors_1018.csv')
    #final_barra_df.to_csv(barra_data_path, index=False)
    #print(f"Final Barra factors saved to: {barra_data_path}")
    #print(final_barra_df.info())

    
    # 5.2 生成并保存行业信息
    # 1. 从你刚创建的 final_barra_df 中获取唯一的股票代码列表
    stk_list_df = final_barra_df[['stocknames']].drop_duplicates().rename(columns={'stocknames': 'ts_code'})

    # 2. 用这个新列表执行你原来的合并操作
    ind_info = pd.merge(stk_list_df, sw_industry_data[['ts_code','l1_code','l1_name','in_date']], on='ts_code', how='left')
    del sw_industry_data # sw_industry_data 已合并
    gc.collect()
    ind_info = ind_info.drop_duplicates(subset=['l1_code', 'l1_name']).rename(
        columns={'l1_code': 'code', 'l1_name': 'industry_names', 'in_date': 'start_date'}
    )[['code', 'industry_names', 'start_date']]
    #industry_info_path = os.path.join(config.RESULT_DIR, 'industry_info_1018.csv')
    #ind_info.to_csv(industry_info_path, index=False)
    #print(f"Industry info saved to: {industry_info_path}")
    

    # --- 5.3 保存结果到MongoDB ---
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        print(f"\nConnected to MongoDB for saving results. Database: '{DB_NAME}'")
        
        # 保存最终的Barra因子数据
        save_df_to_mongodb(db, final_barra_df, 'barra_factors')
        
        # 保存处理后的行业信息
        save_df_to_mongodb(db, ind_info, 'sw_industry_info_for_factors')
        
        client.close()
        print("\nMongoDB saving process finished.")
        
    except Exception as e:
        print(f"\nFailed to connect to MongoDB to save results: {e}")

if __name__ == '__main__':
    main()
