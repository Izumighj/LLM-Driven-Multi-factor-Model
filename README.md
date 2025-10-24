# LLM-Driven-Multi-factor-Model

10.5 更新非线性规模、估值、流动性因子计算框架

10.6 增加盈利、成长、杠杆因子计算框架。框架较为粗糙，后续用wind更好？beta计算框架出现bug。提取了个股季度资产负债表、现金流、财务指标

10.7 如果使用"data/csi300_stk_data_financial_index_balance_cashflow.csv",BETA因子算出来为空，需要用"data/csi300_data_20200101_20250922.csv"数据算beta，之后再合并为fator. 在"data_pre.ipynb"中 对子类因子进行合成，正交化处理。后续需要
进行标准化、缩尾、缺失值填充处理.

10.8 对风格因子中的子类因子进行去极值化，去极值为将2.5倍标准差之外的值，赋值成2.5倍标准差的边界值。添加用申万一级行业指数作为行业因子。然后将数据整理到result/barra_data_csi.csv中。在开源的barra模型中替换数据，成功运行demo.py
