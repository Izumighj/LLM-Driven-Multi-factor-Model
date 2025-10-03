import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from tqdm import tqdm
from functools import reduce
tqdm.pandas()

class FactorCalculator:
    """
    一个用于计算多因子的统一框架。

    使用方法:
    1. 准备好行情数据 (prices_df) 和指数数据 (index_df)。
    2. 实例化计算器: calculator = FactorCalculator(prices_df, index_df)
    3. 运行计算: all_factors_df = calculator.run(['BETA', 'RSTR', 'SIZE', ...])
    """
    COMPOSITE_FACTORS = {
        'VOLATILITY': {
            'components': ['DASTD', 'CMRA', 'HSIGMA'],
            'weights': [0.7, 0.15, 0.15]
        }
    }
    def __init__(self, prices_df: pd.DataFrame, index_df: pd.DataFrame):
        """
        初始化计算器并准备基础数据。

        参数:
        prices_df (pd.DataFrame): 包含所有股票的行情数据。
                                  必须列: ['ts_code', 'trade_date', 'close', 'circ_mv']
                                  'circ_mv' 是计算市值因子所用。
        index_df (pd.DataFrame): 市场指数的行情数据。
                                 必须列: ['trade_date', 'close']
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
        # --- 确保数据类型和顺序 ---
        for df in [self.prices_df, self.index_df]:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        self.prices_df.sort_values(by=['ts_code', 'trade_date'], inplace=True)
        self.index_df.sort_values(by='trade_date', inplace=True)

        # --- 计算日度收益率和对数收益率 ---
        self.prices_df['ret'] = self.prices_df.groupby('ts_code')['close'].transform(pd.Series.pct_change)
        self.prices_df['log_ret'] = self.prices_df.groupby('ts_code')['close'].transform(lambda x: np.log(x) - np.log(x.shift(1)))
        
        self.index_df['market_ret'] = self.index_df['close'].pct_change()

        # --- 合并市场收益率，为计算Beta做准备 ---
        self.master_df = pd.merge(
            self.prices_df,
            self.index_df[['trade_date', 'market_ret']],
            on='trade_date',
            how='left'
        )
        self.master_df.reset_index(inplace=True)
        self.master_df.rename(columns={'index': 'original_index'}, inplace=True)
        print("Data preparation complete.")

    # =================================================================
    # =============== 在下方添加各个因子的计算方法 ======================
    # =================================================================

    def compute_size(self):
        """计算规模因子 (SIZE)"""
        print("Computing SIZE factor...")
        size_df = self.master_df[['original_index', 'ts_code', 'trade_date']].copy()
        # SIZE = ln(流通市值)
        size_df = pd.DataFrame({
            'original_index': self.master_df['original_index'],
            'SIZE': np.log(self.master_df['total_mv'])
        })
        return size_df

    # 在 FactorCalculator 类中
    # 用下面的方法替换掉旧的 compute_beta 方法

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

        # --- 【核心修改】在这里使用关键字参数 (keyword arguments) ---
        results = self.master_df.groupby('ts_code').progress_apply(
            _rolling_beta_hsigma,
            window=T,                # 明确指定 window 参数
            min_p=MIN_PERIODS,       # 明确指定 min_p 参数
            weights_arr=weights,     # 明确指定 weights_arr 参数
            include_groups=False
        )
    
        final_df = results.reset_index().rename(columns={'level_1': 'original_index'})
        return final_df[['original_index', 'BETA', 'HSIGMA']]

    def compute_rstr(self):
        """计算动量因子 (RSTR)"""
        print("Computing RSTR factor...")
        # (我们将之前写的RSTR代码封装到这里)
        T, L, HALF_LIFE, MIN_PERIODS = 504, 21, 126, 42
        WINDOW = T - L
        
        decay = 0.5**(1 / HALF_LIFE)
        weights = decay**np.arange(0, WINDOW)
        
        def _rstr_calc(window_s, weights_arr):
            # ... (此处省略内部函数具体实现, 和之前一样)
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
    # 在 FactorCalculator 类中添加这个新方法
        """计算DASTD因子 (特质波动率)"""
        print("Computing DASTD factor...")
    
        T, HALF_LIFE, MIN_PERIODS = 252, 42, 42
        decay = 0.5**(1 / HALF_LIFE)
        weights = decay**np.arange(T - 1, -1, -1)

        # 1. 先计算超额收益率
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
        dastd_s = self.master_df.groupby('ts_code')['excess_ret'].progress_apply(
            lambda x: pd.Series(
                [_dastd_calc(w, weights) for w in x.rolling(window=T, min_periods=MIN_PERIODS)],
                index=x.index  # <--- 关键补充：确保返回的Series索引正确
            ),
            include_groups=False
        )
    
        # 【核心修改 2】修正 reset_index 和 rename 的逻辑
        # reset_index() 后，值列的列名是 'excess_ret'
        dastd_df = dastd_s.reset_index()
        # 所以我们应该重命名 'excess_ret' 列 和 'level_1' 列
        dastd_df.rename(columns={'excess_ret': 'DASTD', 'level_1': 'original_index'}, inplace=True)
    
        # 现在 dastd_df 中肯定有 'original_index' 和 'DASTD' 了
        return dastd_df[['original_index', 'DASTD']]

    # 在 FactorCalculator 类中添加这个新方法

    def compute_cmra(self):
        """计算CMRA因子 (累计收益范围)"""
        print("Computing CMRA factor...")

        # 12个月 * 21天/月
        T = 252
    
        def _cmra_calc(window_series):
            # 窗口内对数收益率的累积和
            cum_log_ret = window_series.cumsum()
            # 累积收益率 Z(t) 的路径
            z_path = np.exp(cum_log_ret) - 1
        
            max_z = z_path.max()
            min_z = z_path.min()
        
            return np.log(1 + max_z) - np.log(1 + min_z)

        tqdm.pandas(desc="CMRA Calculation per Stock")

        cmra_s = self.master_df.groupby('ts_code')['log_ret'].progress_apply(
            lambda x: pd.Series(
                [_cmra_calc(w) for w in x.rolling(window=T)],
                index=x.index  # <--- 关键补充：确保返回的Series索引正确
            ),
            include_groups=False
        )
    
        cmra_df = cmra_s.reset_index()
        # 所以我们应该重命名 'log_ret' 列
        cmra_df.rename(columns={'log_ret': 'CMRA', 'level_1': 'original_index'}, inplace=True)
    
        return cmra_df[['original_index', 'CMRA']]

    # ... 您可以继续添加 compute_cmra, compute_stom 等等 ...
        
    # =================================================================
    # ===================== 统一运行和结果合并 ========================
    # =================================================================

    def run(self, factors: list):
        """
        运行指定的一系列因子计算并将结果合并。
        (带有tqdm进度条)

        参数:
        factors (list): 包含要计算的因子名称字符串的列表， e.g., ['BETA', 'SIZE']
        
        返回:
        pd.DataFrame: 一个包含所有计算出的因子的大表。
        """
        print(f"\nStarting factor calculation for: {factors}")
        
        factor_methods = {
            'SIZE': self.compute_size,
            'BETA': self.compute_beta_hsigma,
            'RSTR': self.compute_rstr,
            'DASTD': self.compute_dastd,
            'CMRA': self.compute_cmra,
            'HSIGMA': self.compute_beta_hsigma,  # HSIGMA和BETA一起计算
            # ... 在这里继续添加映射 ...
        }
        
        results_list = [self.master_df[['ts_code', 'trade_date', 'original_index']]]
        
        # --- 【核心修改】用tqdm包裹因子列表的循环 ---
        # tqdm会自动创建一个进度条
        # desc参数为进度条提供一个固定的描述
        progress_bar = tqdm(factors, desc="Overall Factor Calculation")
        for factor_name in progress_bar:
            # (可选，但推荐) 动态更新进度条描述，显示当前正在计算的因子
            progress_bar.set_description(f"Processing {factor_name}")
            
            method = factor_methods.get(factor_name.upper())
            if method:
                factor_df = method()
                if factor_df is not None:
                    # 我们只返回 original_index 和因子值，避免列名冲突
                    results_list.append(factor_df)
            else:
                print(f"Warning: Factor '{factor_name}' not found.")
        
        print("\nMerging all factor results...")
        final_df = reduce(lambda left, right: pd.merge(left, right, on='original_index', how='left'), 
                          results_list)
        
        final_df.drop(columns='original_index', inplace=True)
        print("Factor calculation complete.")
        return final_df

# =================================================================
# ========================== 如何使用 =============================
# =================================================================

if __name__ == '__main__':
    # 1. 准备数据

    # 获取当前脚本所在目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    stk_path = os.path.join(BASE_DIR, "csi300_data_20200101_20250922.csv")
    index_path = os.path.join(BASE_DIR, "csi_300_index_20200101_20250930.csv")
    stk_data = pd.read_csv(stk_path)
    index_data = pd.read_csv(index_path)
    

    # 2. 实例化计算器
    calculator = FactorCalculator(prices_df=stk_data, index_df = index_data)
    
    
    # 3. 指定要计算的因子列表，并运行
    factors_to_run = ['BETA', 'RSTR', 'SIZE', 'DASTD', 'CMRA', 'HSIGMA']
    all_factors_df = calculator.run(factors_to_run)
    

    # 4. 查看最终结果
    print("\n--- Final Merged Factors DataFrame ---")
    print(all_factors_df.dropna().tail(15))