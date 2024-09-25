import sys

sys.dont_write_bytecode = True
import pandas as pd
import options as o
import data as d
import portfolios as p


def ew_expected_return(returns_df, span=12):
    """Calculates exponentially weighted expected return.

    Args:
        returns_df: DataFrame of historical asset returns.
        span: Span parameter for the exponential weighting.
        freq: Frequency of the data (e.g., 'M' for monthly, 'D' for daily).

    Returns:
        Float representing the exponentially weighted expected return.
    """

    ew_returns = returns_df.ewm(span=span).mean()
    compounded_growth = (1 + ew_returns).prod()
    n_periods = ew_returns.shape[0]
    return compounded_growth ** (span / n_periods) - 1


def mean_rets(returns_df):
    return returns_df.mean()


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
        p.msr(
            o.cc_cov(r.iloc[win[0] : win[1]]),
            o.annualize_rets(r.iloc[win[0] : win[1]], 12),
        )
        for win in windows
    ]

    weights = pd.concat(msr_weight_bt, axis=1).T
    weights.index = r.iloc[estimation_window:].index
    returns = (weights * r).sum(
        axis="columns", min_count=1
    )  # mincount is to generate NAs if all inputs are NAs
    return returns


test = backtest_ws(returns_df)


test3 = backtest_ws(returns_df)
test4 = pd.concat(test3, axis=1).T
test4 = pd.DataFrame(test4).T
test4.index = returns_df.iloc[12:].index


test1 = pd.concat(test)

test2 = test1["MSR_Sample_Average"]
weights = pd.DataFrame(
    test2, index=returns_df.iloc[12:].index, columns=returns_df.columns
)
returns = (weights * returns_df).sum(
    axis="columns", min_count=1
)  # mincount is to generate NAs if all inputs are NAs
