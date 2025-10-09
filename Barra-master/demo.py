# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:49:20 2019

@author: asus
"""

#import os
#os.chdir('C:\\Users\\asus\\Desktop\\MFM')

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
from mfm.MFM import MFM
import pandas as pd 
import numpy as np


####导入数据


file_path = os.path.join(script_dir, 'data_new', f'barra_data_csi.csv')
data = pd.read_csv(file_path)

naidx = 1*np.sum(pd.isna(data), axis = 1)>0
data = data[~naidx]
data.index = range(len(data))


####行业数据
ind_path = os.path.join(script_dir, 'data_new', f'industry_info.csv')
industry_info = pd.read_csv(ind_path)
industry = np.array([1*(data.industry.values == x) for x in industry_info.code.values]).T
industry = pd.DataFrame(industry, columns = list(industry_info.industry_names.values))
data = pd.concat([data.iloc[:,:4], industry, data.iloc[:,5:]], axis = 1)


model = MFM(data, 28, 10)
(factor_ret, specific_ret, R2) = model.reg_by_time()
nw_cov_ls = model.Newey_West_by_time(q = 2, tao = 252)                 #Newey_West调整
er_cov_ls = model.eigen_risk_adj_by_time(M = 100, scale_coef = 1.4)    #特征风险调整
vr_cov_ls, lamb = model.vol_regime_adj_by_time(tao = 42) 

print("\n\n===================================计算结果预览===================================")

print("\n因子收益率 (前5行):")
print(factor_ret.head())

print("\n最后一次的波动调整协方差矩阵:")
if vr_cov_ls: # 检查列表是否为空
    print(vr_cov_ls[-1])

print("\nR2 (前5行):")
print(R2.head())
    
# --- 【新增功能】保存输出结果 ---
print("\n\n===================================保存计算结果===================================")

# 1. 创建一个用于存放结果的文件夹
output_dir = os.path.join(script_dir, 'results')
os.makedirs(output_dir, exist_ok=True)
print(f"结果将保存到: {output_dir}")

# 2. 保存因子收益率
factor_ret_path = os.path.join(output_dir, 'factor_returns.csv')
factor_ret.to_csv(factor_ret_path)
print(f"- 因子收益率已保存到: {os.path.basename(factor_ret_path)}")

# 3. 保存R平方
r2_path = os.path.join(output_dir, 'r_squared.csv')
R2.to_csv(r2_path)
print(f"- 模型R平方已保存到: {os.path.basename(r2_path)}")

# 4. 合并并保存个股特异性收益率
# specific_ret 是一个DataFrame列表，需要先合并
if specific_ret:
    specific_ret_df = pd.concat(specific_ret)
    specific_ret_path = os.path.join(output_dir, 'specific_returns.csv')
    specific_ret_df.to_csv(specific_ret_path)
    print(f"- 个股特异性收益率已保存到: {os.path.basename(specific_ret_path)}")

# 5. 保存最后一次的因子协方差矩阵
if vr_cov_ls:
    final_cov_matrix = vr_cov_ls[-1]
    final_cov_path = os.path.join(output_dir, 'final_vol_regime_adj_covariance.csv')
    final_cov_matrix.to_csv(final_cov_path)
    print(f"- 最终因子协方差矩阵已保存到: {os.path.basename(final_cov_path)}")

# 6. 保存波动调节乘数 Lambda
if lamb:
    lambda_series = pd.Series(lamb, index=model.sorted_dates, name='lambda')
    lambda_path = os.path.join(output_dir, 'volatility_multiplier_lambda.csv')
    lambda_series.to_csv(lambda_path)
    print(f"- 波动调节乘数Lambda已保存到: {os.path.basename(lambda_path)}")

print("\n所有结果保存完毕。")