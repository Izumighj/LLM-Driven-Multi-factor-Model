# post_processing.py

import pandas as pd
import numpy as np
import statsmodels.api as sm

def winsorize_factors(factor_df: pd.DataFrame, factor_list: list, n_std: float = 2.5) -> pd.DataFrame:
    """
    对指定的因子列进行横截面去极值化（Winsorization）。
    """
    df = factor_df.copy()
    winsorize_func = lambda x: x.clip(
            lower=x.mean() - n_std * x.std(),
            upper=x.mean() + n_std * x.std()
        )
    print(f"\n--- Starting Factor Winsorization (Boundary: Mean ± {n_std} * StdDev) ---")
    for factor in factor_list:
        if factor in df.columns:
            print(f"Processing factor: '{factor}'...")
            df[factor] = df.groupby('trade_date')[factor].transform(winsorize_func)
        else:
            print(f"Warning: Factor '{factor}' not found in DataFrame. Skipping.")
    print("--- Factor Winsorization Complete ---")
    return df

def calculate_composite_factors(factor_df: pd.DataFrame, composite_factor_config: dict) -> pd.DataFrame:
    """
    计算一个或多个大类因子（复合因子）。
    """
    df = factor_df.copy()
    for new_factor, config in composite_factor_config.items():
        print(f"Calculating composite factor: '{new_factor}'...")
        components = config['components']
        weights = np.array(config['weights'])
        numerator = pd.Series(0.0, index=df.index)
        denominator = pd.Series(0.0, index=df.index)
        for i, component_name in enumerate(components):
            if component_name in df.columns:
                numerator += df[component_name].fillna(0) * weights[i]
                denominator += df[component_name].notna() * weights[i]
            else:
                print(f"Warning: Component '{component_name}' not found in DataFrame. Skipping.")
        df[new_factor] = numerator / denominator
    print("\n--- Composite Factor Calculation Complete ---")
    return df

def orthogonalize_factors(factor_df: pd.DataFrame, ortho_config: dict) -> pd.DataFrame:
    """
    对一个或多个因子进行正交化处理。
    """
    df = factor_df.copy()
    def _regression(daily_df: pd.DataFrame, dependent_var: str, independent_vars: list):
        y = daily_df[dependent_var]
        X = daily_df[independent_vars]
        valid_idx = pd.concat([y, X], axis=1).dropna().index
        if len(valid_idx) < len(independent_vars) + 2:
            return pd.Series(np.nan, index=daily_df.index)
        y = y.loc[valid_idx]
        X = sm.add_constant(X.loc[valid_idx])
        model = sm.OLS(y, X).fit()
        return model.resid.reindex(daily_df.index)

    print("\n--- Starting Factor Orthogonalization ---")
    grouped = df.groupby('trade_date')
    for factor_to_ortho, against_factors in ortho_config.items():
        print(f"Orthogonalizing '{factor_to_ortho}' against {against_factors}...")
        residuals = grouped.apply(_regression, dependent_var=factor_to_ortho, independent_vars=against_factors, include_groups=False)
        df[factor_to_ortho] = residuals.reset_index(level=0, drop=True)
    print("--- Factor Orthogonalization Complete ---")
    return df