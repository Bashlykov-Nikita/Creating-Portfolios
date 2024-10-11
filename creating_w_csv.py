import sys

sys.dont_write_bytecode = True

# import logging
import pandas as pd
import numpy as np
import data as d
import options as o
import portfolios as p

# * File for creating .csv files with portfolios weights and backtest

covariance = ["Sample", "CCM", "Shrinkage"]

expected_return = ["Average", "EWA"]

portfolios = ["MSR", "GMV", "EW", "CW", "ERC"]

start_backtest = "2019"


def all_msr(cov_arr: list, er_arr: list) -> pd.DataFrame:
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


def all_ew(r: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the EW portfolio for all assets based on the asset returns "r".

    Args:
        r (DataFrame): DataFrame containing asset returns.

    Returns:
        DataFrame: DataFrame with the EW portfolio weights for all assets.
    """
    all_ew = pd.DataFrame()
    print(f"{portfolios[2]}")
    all_ew.loc[:, f"{portfolios[2]}"] = p.weight_ew(r)
    return all_ew


def all_cw(mct_cap: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cap weight portfolio.

    Args:
        mct_cap (DataFrame): Market capitalization data.

    Returns:
        DataFrame: Portfolio cap weights.
    """
    all_cw = pd.DataFrame()
    print(f"{portfolios[3]}")
    all_cw.loc[:, f"{portfolios[3]}"] = p.weight_cw(mct_cap)
    return all_cw


def all_erc(cov_arr: list) -> pd.DataFrame:
    """
    Calculates the Equal Risk Contributions (ERC) for all portfolios based on the given covariance matrices.

    Args:
        cov_arr (list): List of covariance matrices for each portfolio.

    Returns:
        pandas.DataFrame: DataFrame containing the ERC weights for all portfolios.
    """
    all_erc = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        print(f"{portfolios[4]}_{covariance[i]}")
        all_erc[f"{portfolios[4]}_{covariance[i]}"] = p.equal_risk_contributions(cov)
    return all_erc


def cov_arr(r):
    cov = [o.sample_cov(r), o.cc_cov(r), o.shrinkage_cov(r)]
    return cov


def er_arr(r):
    er = [
        o.annualize_rets(r, 12),
        o.ew_annualized_return(r, 12),
        # o.implied_returns(sigma=cov[0], w=p.weight_cw(mc), delta=2.5),
    ]
    return er


# logger = logging.getLogger(__name__)


def backtest_ws(r, estimation_window=12):
    """
    Backtests the weighted strategies based on the given asset returns.

    Args:
        r (DataFrame): DataFrame containing asset returns.
        estimation_window (int, optional): Number of periods to consider for estimation. Defaults to 12.

    Returns:
        DataFrame: DataFrame with the backtested returns for the weighted strategies.
    """
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
        weight_bt.append(
            pd.concat([all_msr(cov, er), all_gmv(cov), all_ew(r), all_erc(cov)])
        )

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


def weights_csv(index_names: list) -> None:
    """
    Generates csv files for all portfolios based on index names with added logging and error handling.

    Args:
        index_names (list): A list containing index names as str.
    """
    for key in index_names:
        # logger.info(f"Processing index: {key}")
        try:
            r = d.get_returns_df(d.icr_m[f"{key}"])
        except Exception as e:
            # logger.error(f"Error fetching returns data for {key}: {e}")
            continue

        try:
            mc = d.get_mkt_cap(d.icr_m[f"{key}"])
        except Exception as e:
            # logger.error(f"Error fetching market cap data for {key}: {e}")
            continue

        cov = cov_arr(r)
        er = er_arr(r)
        pd.concat(
            [
                all_msr(cov, er),
                all_gmv(cov),
                all_ew(r),
                all_cw(mc),
                all_erc(cov),
            ],
            axis=1,
        ).to_csv(f"portfolios_data/{key}_portfolios.csv", index=True)
        print(f"backtest_portfolios_data {key}")
        backtest_ws(r[start_backtest:]).to_csv(
            f"backtest_portfolios_data/{key}_backtest_portfolios.csv", index=True
        )


weights_csv(d.index_names)
