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
        pandas.DataFrame: DataFrame containing the maximum sharpe ratio for each combination
    """
    all_msr = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        for j, er in enumerate(er_arr):
            print(f"{portfolios[0]}_{covariance[i]}_{expected_return[j]}")
            all_msr[f"{portfolios[0]}_{covariance[i]}_{expected_return[j]}"] = p.msr(
                cov, er
            )
    return all_msr


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
    msr_weight_bt = [
        all_msr(cov_arr(r.iloc[win[0] : win[1]]), er_arr(r.iloc[win[0] : win[1]]))
        for win in windows
    ]

    apply_weights = []
    for i, df in enumerate(msr_weight_bt):
        apply_weights.append(
            df.apply(lambda x: x * r.iloc[i + estimation_window], axis=0).sum(
                axis="rows"
            )
        )
    returns = pd.concat(apply_weights, axis=1).T
    returns.index = r.index[estimation_window:]
    return returns


test = backtest_ws(returns_df)
