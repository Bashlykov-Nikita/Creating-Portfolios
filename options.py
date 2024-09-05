import numpy as np
import pandas as pd
import data


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std() * (periods_per_year**0.5)


# Constant-Correlation Model (Elton-Gruber)
covariance = ["Sample", "CCM", "Shrinage"]
# Rlling Window,
# Exponentialy Weighted Moving Avarage,
# Black-Litterman Model
expacted_return = ["Average", "RW", "EWMA", "BLM"]


def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()


def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.0)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)


def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta * prior + (1 - delta) * sample


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1
