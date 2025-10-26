# factor_calculator.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
#from functools import reduce
import gc
tqdm.pandas()

class FactorCalculator:
    """
    一个用于计算多因子的统一框架。

    1. 行情数据 (prices_df) 和指数数据 (index_df)。
    2. calculator = FactorCalculator(prices_df, index_df)
    3. 运行计算: all_factors_df = calculator.run(['BETA', 'RSTR', 'SIZE', ...])
    4. 结果得到barra_data.csv和industry_info.csv用于barra demo的计算
    """

    def __init__(self, prices_df: pd.DataFrame, index_df: pd.DataFrame):
        """
        初始化计算器并准备基础数据。

        参数:
        prices_df (pd.DataFrame): 包含所有股票的行情数据。
        index_df (pd.DataFrame): 市场指数的行情数据。
        """
        print("Initializing Factor Calculator...")
        self.prices_df = prices_df
        self.index_df = index_df
        self._prepare_data()

    def _prepare_data(self):
        """
        数据预处理：排序、计算收益率等，避免重复计算。
        """
        print("Preparing base data (returns, etc.)...")
        # --- 日期转换 "2020/01/01" 正序排列 ---
        #self.index_df['trade_date'] = pd.to_datetime(self.index_df['trade_date'],format='%Y%m%d')
        self.index_df['trade_date'] = self.index_df['trade_date'].dt.strftime('%Y/%m/%d')

        #self.prices_df['trade_date'] = pd.to_datetime(self.prices_df['trade_date'])
        self.prices_df['trade_date'] = self.prices_df['trade_date'].dt.strftime('%Y/%m/%d')
        
        self.prices_df.sort_values(by=['ts_code', 'trade_date'], inplace=True)
        self.index_df.sort_values(by='trade_date', inplace=True)

        # --- 计算日度收益率和对数收益率 ---
        self.prices_df['ret'] = self.prices_df.groupby('ts_code')['close'].transform(pd.Series.pct_change)
        self.prices_df['log_ret'] = self.prices_df.groupby('ts_code')['close'].transform(lambda x: np.log(x) - np.log(x.shift(1)))
        
        self.index_df['market_ret'] = self.index_df['close'].pct_change()

        # --- 合并市场收益率 ---
        self.master_df = pd.merge(
            self.prices_df,
            self.index_df[['trade_date', 'market_ret']],
            on='trade_date',
            how='left'
        )
        self.master_df.reset_index(inplace=True)
        self.master_df.rename(columns={'index': 'original_index'}, inplace=True)
        print("Data preparation complete.")

    # =============== 各个因子的计算方法 ======================

    def compute_size(self):
        """计算规模因子 (SIZE)"""
        print("Computing SIZE factor...")
        size_df = self.master_df[['original_index', 'ts_code', 'trade_date']].copy()
        # SIZE = ln(总市值)
        size_df = pd.DataFrame({
            'original_index': self.master_df['original_index'],
            'SIZE': np.log(self.master_df['total_mv'])
        })
        return size_df

    def compute_beta_hsigma(self):
        """
        计算BETA和HSIGMA因子。
        HSIGMA是Beta回归残差的标准差。
        """
        print("Computing BETA and HSIGMA factors...")
    
        T, HALF_LIFE, MIN_PERIODS = 252, 63, 42
        decay = 0.5**(1 / HALF_LIFE)
        weights = decay**np.arange(T - 1, -1, -1)
    
        def _beta_hsigma_calc(window_df, weights_arr):
            df = window_df.dropna()
            if df.shape[0] < MIN_PERIODS:
                return pd.Series({'BETA': np.nan, 'HSIGMA': np.nan})

            y = df['ret']
            X = sm.add_constant(df['market_ret'])
            current_weights = weights_arr[-df.shape[0]:]
        
            model = sm.WLS(y, X, weights=current_weights).fit()
        
            beta = model.params.get('market_ret', np.nan)
            hsigma = np.sqrt(model.scale)
        
            return pd.Series({'BETA': beta, 'HSIGMA': hsigma})

        def _rolling_beta_hsigma(stock_df, window, min_p, weights_arr):
            returns = stock_df[['ret', 'market_ret']]
            windows = returns.rolling(window=window, min_periods=min_p)
            results_df = pd.concat([_beta_hsigma_calc(w, weights_arr) for w in windows], axis=1).T
            results_df.index = stock_df.index
            return results_df

        tqdm.pandas(desc="Beta/Hsigma per Stock")

        stk_df = self.master_df[['trade_date','ts_code','ret','market_ret']]
        results = stk_df.groupby('ts_code').progress_apply(
            _rolling_beta_hsigma,
            window=T,                # window 参数
            min_p=MIN_PERIODS,       # min_p 参数
            weights_arr=weights,     # weights_arr 参数
            include_groups=False
        )
    
        final_df = results.reset_index().rename(columns={'level_1': 'original_index'})
        return final_df[['original_index', 'BETA', 'HSIGMA']]

    def compute_rstr(self):
        """计算动量因子 (RSTR)"""
        print("Computing RSTR factor...")
        T, L, HALF_LIFE, MIN_PERIODS = 504, 21, 126, 42
        WINDOW = T - L
        
        decay = 0.5**(1 / HALF_LIFE)
        weights = decay**np.arange(0, WINDOW)
        
        def _rstr_calc(window_s, weights_arr):
            ws = pd.Series(weights_arr[:len(window_s)], index=window_s.index)
            valid_ret = window_s.dropna()
            if len(valid_ret) < MIN_PERIODS: return np.nan
            valid_w = ws.loc[valid_ret.index]
            norm_w = valid_w / np.sum(valid_w)
            return np.sum(valid_ret * norm_w)

        tqdm.pandas(desc="RSTR Calculation per Stock")
        
        rstr_s = self.master_df.groupby('ts_code')['log_ret'].progress_apply(
            lambda x: x.shift(L).rolling(window=WINDOW, min_periods=MIN_PERIODS)
                         .apply(_rstr_calc, args=(weights,), raw=False),
            include_groups=False
        )
        
        rstr_df = rstr_s.reset_index().rename(columns={'log_ret': 'RSTR', 'level_1': 'original_index'})
        return rstr_df[['original_index', 'RSTR']]
    
    def compute_dastd(self):
        """计算DASTD因子 (特质波动率)"""
        print("Computing DASTD factor...")
    
        T, HALF_LIFE, MIN_PERIODS = 252, 42, 42
        decay = 0.5**(1 / HALF_LIFE)
        weights = decay**np.arange(T - 1, -1, -1)

        # 先计算超额收益率
        self.master_df['excess_ret'] = self.master_df['ret'] - self.master_df['market_ret']
    
        def _dastd_calc(window_series, weights_arr):
            valid_ret = window_series.dropna()
            if len(valid_ret) < MIN_PERIODS:
                return np.nan
        
            # 权重对齐和归一化
            ws = pd.Series(weights_arr[-len(valid_ret):], index=valid_ret.index)
            norm_w = ws / np.sum(ws)
        
            # 计算加权均值
            weighted_mean = np.sum(valid_ret * norm_w)
            # 计算加权方差
            weighted_var = np.sum(norm_w * ((valid_ret - weighted_mean) ** 2))
            # 返回加权标准差
            return np.sqrt(weighted_var)

        tqdm.pandas(desc="DASTD Calculation per Stock")
        # <优化：将Python列表循环改为Pandas的 rolling().apply() ---
        dastd_s = self.master_df.groupby('ts_code')['excess_ret'].progress_apply(
            lambda x: x.rolling(window=T, min_periods=MIN_PERIODS)
                         .apply(_dastd_calc, args=(weights,), raw=False),
            include_groups=False
        )
        # --- 结束修改 ---
    
        dastd_df = dastd_s.reset_index()
        dastd_df.rename(columns={'excess_ret': 'DASTD', 'level_1': 'original_index'}, inplace=True)
        # 清理临时列
        self.master_df.drop(columns=['excess_ret'], inplace=True)
    
        return dastd_df[['original_index', 'DASTD']]


    def compute_cmra(self):
        """计算CMRA因子 (累计收益范围)"""
        print("Computing CMRA factor...")

        # 12个月 * 21天/月
        T = 252
    
        def _cmra_calc(window_series):
            # 优化: 确保窗口期足够 ---
            if window_series.shape[0] < T:
                 return np.nan
            # --- 结束修改 ---
            # 窗口内对数收益率的累积和
            cum_log_ret = window_series.cumsum()
            # 累积收益率 Z(t) 的路径
            z_path = np.exp(cum_log_ret) - 1
        
            max_z = z_path.max()
            min_z = z_path.min()
        
            return np.log(1 + max_z) - np.log(1 + min_z)

        tqdm.pandas(desc="CMRA Calculation per Stock")

        # 优化：将Python列表循环改为Pandas的 rolling().apply() ---
        cmra_s = self.master_df.groupby('ts_code')['log_ret'].progress_apply(
            lambda x: x.rolling(window=T)
                         .apply(_cmra_calc, raw=False), # raw=False 因为 _cmra_calc 需要 pandas Series
            include_groups=False
        )
        # --- 结束修改 ---
    
        cmra_df = cmra_s.reset_index()
        cmra_df.rename(columns={'log_ret': 'CMRA', 'level_1': 'original_index'}, inplace=True)
    
        return cmra_df[['original_index', 'CMRA']]


    def compute_nlsize(self):
        """
        计算非线性规模因子 (NonLinear Size)。
        因子定义：将股票总市值对数的三次方对总市值对数进行横截面回归，取残差的相反数。
        """
        print("Computing NonLinear Size (NLSIZE) factor...")

        # 1. 确保基础的SIZE因子存在
        if 'SIZE' not in self.master_df.columns:
            self.master_df['SIZE'] = np.log(self.master_df['total_mv'])

        # 2. 计算SIZE因子的三次方
        self.master_df['SIZE_CUBE'] = self.master_df['SIZE']**3

        # 定义用于横截面回归的函数
        def _nlsize_regression(daily_df: pd.DataFrame) -> pd.Series:
            """对每日的截面数据进行回归"""
            # 移除缺失值
            valid_data = daily_df[['SIZE', 'SIZE_CUBE']].dropna()

            # 如果有效数据不足以进行回归，则返回空值
            if valid_data.shape[0] < 2:
                return pd.Series(np.nan, index=daily_df.index)

            # 准备回归的X和y
            y = valid_data['SIZE_CUBE']
            X = sm.add_constant(valid_data['SIZE'])

            # 执行OLS回归
            model = sm.OLS(y, X).fit()

            # 获取残差
            residuals = pd.Series(model.resid, index=valid_data.index)

            # 将残差匹配回原始索引，并取其相反数
            # 对于没有参与回归的行（因为是NaN），结果也是NaN
            nlsize = -residuals.reindex(daily_df.index)

            return nlsize

        tqdm.pandas(desc="NLSIZE Cross-sectional Regression")
    
        # 3. 按交易日分组，对每个截面应用回归函数
        nlsize_s = self.master_df.groupby('trade_date').progress_apply(_nlsize_regression)
    
        # 4. 格式化结果并返回
        nlsize_s = nlsize_s.reset_index(level=0, drop=True).sort_index()

        nlsize_df = pd.DataFrame({
            'original_index': self.master_df['original_index'],
            'NLSIZE': nlsize_s
        })

        # 清理临时列
        self.master_df.drop(columns=['SIZE', 'SIZE_CUBE'], inplace=True, errors='ignore')

        return nlsize_df[['original_index', 'NLSIZE']]
    
    def compute_bp(self):
        """
        计算估值因子 BP (Book-to-Price)。
        因子定义：市净率(PB)的倒数。
        """
        print("Computing Value (BP) factor...")

        # 检查必需的 'pb' 列是否存在
        if 'pb' not in self.master_df.columns:
            print("\nERROR: 'pb' column not found in the input data.")
            print("To compute the BP factor, you must provide Price-to-Book ratio data.\n")
            return None  # 返回None以中断计算

        # 提取PB序列
        pb_series = self.master_df['pb']
    
        # 计算BP = 1 / PB
        # 当市净率小于等于0时，其倒数无经济意义，我们将其设为缺失值(NaN)
        bp_values = np.where(pb_series > 0, 1 / pb_series, np.nan)

        # 构建并返回结果DataFrame
        bp_df = pd.DataFrame({
            'original_index': self.master_df['original_index'],
            'BP': bp_values
        })
    
        return bp_df[['original_index', 'BP']]


    def compute_liquidity(self):
        """
        计算原始的流动性因子 STOM, STOQ, STOA。
        它们分别是月度、季度、年度的对数累计换手率。
        """
        print("Computing base Liquidity factors (STOM, STOQ, STOA)...")

        # 1. 检查必需的 'turnover_rate' 列是否存在
        if 'turnover_rate' not in self.master_df.columns:
            print(f"\nERROR: 'turnover_rate' column not found in the input data.\n")
            return None

        # 2. 获取日度换手率，并将其从百分比转换成比率
        # turnover_rate' 的单位是百分比 (e.g., 1.5 代表 1.5%)，所以除以100
        dtv = self.master_df['turnover_rate'] / 100.0
    
        # 3. 按股票分组，计算滚动累计换手率
        grouped_dtv = dtv.groupby(self.master_df['ts_code'])

    
        # <--- 3. 优化：移除多余的 progress_apply，直接在 groupby.rolling 上计算 ---
        print("Calculating STOM...")
        stom_base = grouped_dtv.rolling(window=21, min_periods=15).sum()
        print("Calculating STOQ...")
        stoq_base = grouped_dtv.rolling(window=63, min_periods=42).sum()
        print("Calculating STOA...")
        stoa_base = grouped_dtv.rolling(window=252, min_periods=126).sum()
    
        # 结果是 MultiIndex, 我们需要去掉 'ts_code' 级别以匹配 'original_index'
        stom = np.log(stom_base.droplevel(0).replace(0, np.nan))
        stoq = np.log(stoq_base.droplevel(0).replace(0, np.nan))
        stoa = np.log(stoa_base.droplevel(0).replace(0, np.nan))
        # --- 结束修改 ---

        # 构建并返回结果 DataFrame
        liquidity_df = pd.DataFrame({
            'original_index': self.master_df.index,
            'STOM': stom.values,
            'STOQ': stoq.values,
            'STOA': stoa.values
        })


        return liquidity_df
    


    def compute_earnings_yield(self):
        """
        计算盈利因子 CETOP 和 ETOP。
        此方法现在内部处理从单季度现金流到TTM的计算。
        - CETOP: 经营现金流TTM / 总市值
        - ETOP: 1 / PE_ttm (市盈率倒数)
        """
        print("Computing Earnings Yield factors (CETOP, ETOP)...")

        # 1. 检查所需的原始数据列
        # CETOP 现在需要 'n_cashflow_act' (单季度), 'end_date', 和 'total_mv'
        # ETOP 需要 'pe_ttm'
        required_cols = ['n_cashflow_act', 'end_date', 'total_mv', 'pe_ttm']
        if not all(col in self.master_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.master_df.columns]
            print(f"\nERROR: Missing required columns for earnings yield calculation. "
                  f"Please ensure your merged data contains: {missing}\n")
            return None

        # --- CETOP 计算 ---
        # 2. 从主表中提取唯一的财务报告期数据，以避免重复计算
        financial_data = self.master_df[['ts_code', 'end_date', 'n_cashflow_act']].drop_duplicates().copy()

        # 3. 按股票和报告期排序，为滚动计算做准备
        financial_data.sort_values(by=['ts_code', 'end_date'], inplace=True)
        
        # 4. 计算TTM值：按股票分组，对最近4个季度的现金流求和
        financial_data['n_cashflow_act_ttm'] = financial_data.groupby('ts_code')['n_cashflow_act'].transform(
            lambda x: x.rolling(window=4, min_periods=4).sum()
        )

        # 优化：不修改 self.master_df，而是创建一个临时DF ---
        temp_df = pd.merge(
            self.master_df[['original_index', 'ts_code', 'end_date', 'total_mv', 'pe_ttm']],
            financial_data[['ts_code', 'end_date', 'n_cashflow_act_ttm']],
            on=['ts_code', 'end_date'],
            how='left'
        )
        # 确保索引与 master_df 一致 (merge后可能乱序)
        temp_df.sort_values(by='original_index', inplace=True)

        cash_flow_ttm = temp_df['n_cashflow_act_ttm']
        total_mv = temp_df['total_mv']
        cetop_values = np.where(
            (total_mv > 0) & (cash_flow_ttm > 0),
            cash_flow_ttm / total_mv,
            np.nan
        )
    
        # --- ETOP 计算 (逻辑不变) ---
        pe_series = temp_df['pe_ttm']
        etop_values = np.where(pe_series > 0, 1 / pe_series, np.nan)
    
        earnings_df = pd.DataFrame({
            'original_index': temp_df['original_index'],
            'CETOP': cetop_values,
            'ETOP': etop_values
        })
        
        del temp_df # 清理临时DF
        gc.collect()
        # --- 结束修改 ---
    
        return earnings_df

    def select_growth_factors(self):
        """
        从主数据表中选取已经过时点对齐的成长因子。
        - YOYProfit: q_profit_yoy (单季度净利同比增长率)
        - YOYSales: q_sales_yoy (单季度营收同比增长率)
        """
        print("Selecting Growth factors (YOYProfit, YOYSales)...")

        # 1. 检查必需的原始列是否存在
        required_cols = ['q_profit_yoy', 'q_sales_yoy']
        if not all(col in self.master_df.columns for col in required_cols):
            print(f"\nERROR: Missing required columns for growth factors. "
                f"Please ensure your merged data contains: {required_cols}\n")
            return None

        # 2. 构建并返回结果 DataFrame
        # 直接选取并重命名列
        growth_df = self.master_df[['original_index', 'q_profit_yoy', 'q_sales_yoy']].copy()
        #原序列是百分比，需要转化为比值
        growth_df['q_profit_yoy'] = growth_df['q_profit_yoy']/100
        growth_df['q_sales_yoy'] = growth_df['q_sales_yoy']/100
        growth_df.rename(columns={
            'q_profit_yoy': 'YOYProfit',
            'q_sales_yoy': 'YOYSales'
        }, inplace=True)
    
        return growth_df

    def compute_leverage(self):
        """
        计算杠杆因子 MLEV, DTOA, BLEV。
        此方法要求 master_df 中已包含对齐好的日度基本面数据。
        """
        print("Computing Leverage factors (MLEV, DTOA, BLEV)...")

        # 1. 定义并检查所需的列名
        required_cols = [
            'total_mv', 'total_ncl', 
            'total_hldr_eqy_inc_min_int', 'debt_to_assets'
        ]
        if not all(col in self.master_df.columns for col in required_cols):
            print(f"\nERROR: Missing required columns for leverage calculation. "
                f"Please ensure your merged data contains: {required_cols}\n")
            return None

        # 2. 计算各个杠杆因子

        # MLEV = (总市值 + 非流动负债) / 总市值
        # total_mv = total_mv, total_ncl = 非流动负债
        mlev = (self.master_df['total_mv'] + self.master_df['total_ncl']) / self.master_df['total_mv']
        mlev.replace([np.inf, -np.inf], np.nan, inplace=True) # 处理总市值为0的特殊情况

        # DTOA = 资产负债率
        dtoa = self.master_df['debt_to_assets']
    
        # BLEV = (账面价值 + 非流动负债) / 账面价值
        # total_hldr_eqy_inc_min_int = 账面价值
        book_value = self.master_df['total_hldr_eqy_inc_min_int']
        # 当账面价值为负或为0时，该因子无经济意义，设为NaN
        blev = np.where(
            book_value > 0,
            (book_value + self.master_df['total_ncl']) / book_value,
            np.nan
        )
    
        # 3. 构建并返回结果 DataFrame
        leverage_df = pd.DataFrame({
            'original_index': self.master_df['original_index'],
            'MLEV': mlev,
            'DTOA': dtoa,
            'BLEV': blev
        })
    
        return leverage_df
        
    # =================================================================
    # ===================== 统一运行和结果合并 ========================
    # =================================================================

    def run(self, factors: list):
        """
        参数:
        factors (list): 包含要计算的因子名称字符串的列表， e.g., ['BETA', 'SIZE']
        
        返回:
        pd.DataFrame: 一个包含所有计算出的因子的大表。
        """
        print(f"\nStarting factor calculation for: {factors}")
        
        factor_methods = {
            'SIZE': self.compute_size,
            'BETA': self.compute_beta_hsigma,  # HSIGMA和BETA一起计算
            'RSTR': self.compute_rstr,
            'DASTD': self.compute_dastd,
            'CMRA': self.compute_cmra,
            'NLSIZE': self.compute_nlsize,
            'BP': self.compute_bp,
            'LIQUIDITY': self.compute_liquidity,
            'EARNINGS':self.compute_earnings_yield,
            'GROWTH': self.select_growth_factors,
            'LEVERAGE': self.compute_leverage
            # ... 在这里继续添加映射 ...
        }
        
        
        
        # <--- 5. 【核心修改】优化合并策略 ---
        # 不再使用 results_list 和 reduce
        
        # 1. 初始化一个基础DataFrame
        final_df = self.master_df[['ts_code', 'trade_date', 'original_index', 'ret', 'circ_mv']].copy()
        
        # 2. 迭代计算和合并
        print("\nIteratively computing and merging factors...")
        for factor_name in tqdm(factors, desc="Overall Factor Calculation"):
            method = factor_methods.get(factor_name.upper())
            if not method:
                print(f"Warning: Factor '{factor_name}' not found.")
                continue
                
            # 计算因子
            factor_df = method()
            
            if factor_df is not None:
                # 立即合并
                final_df = pd.merge(
                    final_df, 
                    factor_df, 
                    on='original_index', 
                    how='left'
                )
                # 立即删除中间表并回收内存
                del factor_df
                gc.collect()
            else:
                print(f"Warning: Method for '{factor_name}' returned None.")
        
        # --- 结束修改 ---
        
        final_df.drop(columns='original_index', inplace=True)
        print("Factor calculation complete.")
        return final_df