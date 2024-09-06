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
# Rolling Window,
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


# TODO:
# def RW()
# def EWRA()


# Black-Litterman:
def as_colvec(x):
    if x.ndim == 2:
        return x
    else:
        return np.expand_dims(x, axis=1)


def implied_returns(delta, sigma, w):
    """
    Obtain the implied expected returns by reverse engineering the weights
    Inputs:
    delta: Risk Aversion Coefficient (scalar)
    sigma: Variance-Covariance Matrix (N x N) as DataFrame
        w: Portfolio weights (N x 1) as Series
    Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze()  # to get a series from a 1-column dataframe
    ir.name = "Implied Returns"
    return ir


# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(
        np.diag(np.diag(helit_omega.values)), index=p.index, columns=p.index
    )


from numpy.linalg import inv


def bl(w_prior, sigma_prior, p, q, omega=None, delta=2.5, tau=0.02):
    """
    # Computes the posterior expected returns based on
    # the original black litterman reference model
    #
    # W.prior must be an N x 1 vector of weights, a Series
    # Sigma.prior is an N x N covariance matrix, a DataFrame
    # P must be a K x N matrix linking Q and the Assets, a DataFrame
    # Q must be an K x 1 vector of views, a Series
    # Omega must be a K x K matrix a DataFrame, or None
    # if Omega is None, we assume it is
    #    proportional to variance of the prior
    # delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior, w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(
        inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values)
    )
    # posterior estimate of uncertainty of mu.bl
    #     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = (
        sigma_prior
        + sigma_prior_scaled
        - sigma_prior_scaled.dot(p.T)
        .dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega))
        .dot(p)
        .dot(sigma_prior_scaled)
    )
    return (mu_bl, sigma_bl)
