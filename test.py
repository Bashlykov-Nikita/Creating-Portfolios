import numpy as np
import sys

sys.dont_write_bytecode = True
import pandas as pd
import options as o
import data as d
import portfolios as p

# Example usage
returns_df = d.get_returns_df(d.icr_m["DAX"])

portfolios = ["MSR", "GMV", "EW", "CW", "ERC"]
covariance = ["Sample", "CCM", "Shrinage"]
expected_return = ["Average", "EW"]


def all_msr(cov_arr, er_arr):
    """
    Calculates the maximum sharpe ratio for all combinations of covariance matrices and expected returns.

    Args:
        cov_arr (array-like): Array of covariance matrices
        er_arr (array-like): Array of expected returns

    Returns:
        numpy.ndarray: Array containing the maximum sharpe ratio for each combination
    """
    all_msr = np.empty((len(cov_arr), len(er_arr)))
    for i, cov in enumerate(cov_arr):
        for j, er in enumerate(er_arr):
            all_msr[i, j] = p.msr(cov, er)
    return all_msr


def all_gmv(cov_arr: list) -> pd.DataFrame:
    """
    Calculate the Global Minimum Volatility (GMV) portfolio weights for each covariance matrix in the array.

    Args:
        cov_arr (list): List of covariance matrices for which to calculate GMV portfolio weights.

    Returns:
        pandas.DataFrame: DataFrame containing GMV portfolio weights for each covariance matrix.
    """
    all_gmv = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        print(f"{portfolios[1]}_{covariance[i]}")
        all_gmv[f"{portfolios[1]}_{covariance[i]}"] = p.gmv(cov)
    return all_gmv


def cov_arr(r):
    cov = [o.sample_cov(r), o.cc_cov(r), o.shrinkage_cov(r)]
    return cov


def er_arr(r):
    er = [
        o.annualize_rets(r, 12),
        o.ew_annualized_return(r, 12),
        # o.implied_returns(sigma=cov[0], w=p.weight_cw(cm), delta=2.5),
    ]
    return er


def backtest_ws(r, estimation_window=12):
    n_periods = r.shape[0]
    # return windows
    windows = [
        (start, start + estimation_window)
        for start in range(n_periods - estimation_window)
    ]
    weight_bt = []
    for win in windows:
        cov = cov_arr(r.iloc[win[0] : win[1]])
        er = er_arr(r.iloc[win[0] : win[1]])
        weight_bt.append(pd.concat([all_msr(cov, er), all_gmv(cov)]))

    apply_weights = []
    for i, df in enumerate(weight_bt):
        apply_weights.append(
            df.apply(lambda x: x * r.iloc[i + estimation_window], axis=0).sum(
                axis="rows"
            )
        )
    returns = pd.concat(apply_weights, axis=1).T
    returns.index = r.index[estimation_window:]
    return returns


test = backtest_ws(returns_df)
