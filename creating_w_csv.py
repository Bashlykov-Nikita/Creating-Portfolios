import logging
import pandas as pd
import numpy as np
import data as d
import options as o
import portfolios as p

covariance = ["Sample", "CCM", "Shrinage"]

expected_return = ["Average", "Implied"]

portfolios = ["MSR", "GMV", "EW", "CW", "ERC"]


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


def all_gmv(cov_arr):
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


def all_ew(r):
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


def all_cw(mct_cap):
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


def all_erc(cov_arr):
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


logger = logging.getLogger(__name__)


def weights_csv(index_names):
    """
    Generates csv files for all portfolios based on index names with added logging and error handling.

    Args:
        index_names (dict): A dictionary containing index names as keys.
    """
    for key in index_names:
        logger.info(f"Processing index: {key}")
        try:
            r = d.get_returns_df(d.icr_m[f"{key}"])
        except Exception as e:
            logger.error(f"Error fetching returns data for {key}: {e}")
            continue

        try:
            cm = d.get_mkt_cap(d.icr_m[f"{key}"])
        except Exception as e:
            logger.error(f"Error fetching market cap data for {key}: {e}")
            continue

        cov = [o.sample_cov(r), o.cc_cov(r), o.shrinkage_cov(r)]
        er = [
            o.annualize_rets(r, 12),
            o.implied_returns(sigma=cov[0], w=p.weight_cw(cm), delta=2.5),
        ]
        pd.concat(
            [
                all_msr(cov, er),
                all_gmv(cov),
                all_ew(r),
                all_cw(cm),
                all_erc(cov),
            ],
            axis=1,
        ).to_csv(f"{key}_portfolios", index=True)


weights_csv(d.index_names)
