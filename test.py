import pandas as pd
import numpy as np
import data as d
import options as o
import portfolios as p


covariance = ["Sample", "CCM", "Shrinage"]

expected_return = ["Average", "Implied"]

portfolios = ["MSR", "GMV", "EW", "CW", "ERC"]

r = d.get_returns_df(d.icr_m["NasdaqComposite"])
cm = d.get_mkt_cap(d.icr_m["NasdaqComposite"])
cov_arr = [o.sample_cov(r), o.cc_cov(r), o.shrinkage_cov(r)]
er_arr = [
    o.annualize_rets(r, 12),
    o.implied_returns(sigma=cov_arr[0], w=p.weight_cw(cm), delta=2.5),
]


def all_msr(cov_arr, er_arr):
    all_msr = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        for j, er in enumerate(er_arr):
            print(f"{portfolios[0]}_{covariance[i]}_{expected_return[j]}")
            all_msr[f"{portfolios[0]}_{covariance[i]}_{expected_return[j]}"] = p.msr(
                cov, er
            )
    return all_msr


def all_gmv(cov_arr):
    all_gmv = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        print(f"{portfolios[1]}_{covariance[i]}")
        all_gmv.insert(i, f"{portfolios[1]}_{covariance[i]}", p.gmv(cov))
    return all_gmv


def all_ew(r):
    all_ew = pd.DataFrame()
    print(f"{portfolios[2]}")
    all_ew.loc[:, f"{portfolios[2]}"] = p.weight_ew(r)
    return all_ew


def all_cw(mct_cap):
    all_cw = pd.DataFrame()
    print(f"{portfolios[3]}")
    all_cw.loc[:, f"{portfolios[3]}"] = p.weight_cw(mct_cap)
    return all_cw


def all_erc(cov_arr):
    all_erc = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        print(f"{portfolios[4]}_{covariance[i]}")
        all_erc.insert(
            i, f"{portfolios[4]}_{covariance[i]}", p.equal_risk_contributions(cov)
        )
    return all_erc


nasdaq_portfolios = pd.DataFrame()
nasdaq_portfolios = pd.concat(
    [
        all_msr(cov_arr, er_arr),
        all_gmv(cov_arr),
        all_ew(r),
        all_cw(cm),
        all_erc(cov_arr),
    ],
    axis=1,
)
