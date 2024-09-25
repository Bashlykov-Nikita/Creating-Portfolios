import pandas as pd
import options as o
import data as d


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
returns_df = d.get_returns_df(d.icr_m["HSI"])


mr = mean_rets(returns_df)
ew_er = ew_expected_return(returns_df)
print(ew_er)
a_r = o.annualize_rets(returns_df, 12)
print(abs(ew_er - a_r))
