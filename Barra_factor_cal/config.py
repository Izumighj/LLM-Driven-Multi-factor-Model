# config.py

import os

# 1. 文件路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")

STK_DATA_PATH = os.path.join(DATA_DIR, "csi300_stk_data_financial_index_balance_cashflow.csv")
INDEX_DATA_PATH = os.path.join(DATA_DIR, "csi_300_index_20200101_20250930.csv")
INDUSTRY_DATA_PATH = os.path.join(DATA_DIR, "stk_sw_industry.csv")



# 2. 因子计算配置
FACTORS_TO_RUN = [
    'SIZE', 'BETA', 'RSTR', 'DASTD', 'CMRA', 'NLSIZE', 'BP', 
    'LIQUIDITY', 'EARNINGS', 'GROWTH', 'LEVERAGE'
]

# 3. 因子合成配置 (Composite Factor Rules)
COMPOSITE_CONFIG = {
    'volatility': {
        'components': ['DASTD', 'CMRA', 'HSIGMA'],
        'weights': [0.7, 0.15, 0.15]
    },
    'leverage': {
        'components': ['MLEV', 'DTOA', 'BLEV'],
        'weights': [1/3, 1/3, 1/3]
    },
    'liquidity': {
        'components': ['STOM', 'STOQ', 'STOA'],
        'weights': [0.5, 0.25, 0.25]
    },
    'earnings': {
        'components': ['CETOP', 'ETOP'],
        'weights': [0.5, 0.5]
    },
    'growth': {
        'components': ['YOYProfit', 'YOYSales'],
        'weights': [0.5, 0.5]
    }
}

# 4. 因子正交化处理
ORTHO_RULES = {
    'volatility': ['BETA', 'SIZE'],
    'liquidity': ['SIZE']
}

# 5. Barra风格列名重命名
COLUMN_RENAME_MAP = {
    'trade_date': 'date',
    'ts_code': 'stocknames',
    'l1_code': 'industry',
    'circ_mv': 'capital',
    'SIZE': 'size',
    'BETA': 'beta',
    'RSTR': 'momentum',
    'volatility': 'residual_volatility',
    'NLSIZE': 'non_linear_size',
    'BP': 'book_to_price_ratio',
    'earnings': 'earnings_yield',
}

# 6. Barra最终输出列顺序
BARRA_OUTPUT_COLUMNS = [
    'date', 'stocknames', 'capital', 'ret', 'industry',
    'size', 'beta', 'momentum', 'residual_volatility', 'non_linear_size',
    'book_to_price_ratio', 'liquidity', 'earnings_yield', 'growth', 'leverage'
]