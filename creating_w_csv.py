import pandas as pd
import numpy as np
import data as d
import options as o
import portfolios as p

covariance = ["Sample", "CCM", "Shrinage"]

expacted_return = ["Average", "Implied"]

portfolios = ["MSR", "GMV", "EW", "CW", "ERC"]


def all_msr(cov_arr, er_arr):
    all_msr = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        for j, er in enumerate(er_arr):
            print(f"{portfolios[0]}_{covariance[i]}_{expacted_return[j]}")
            all_msr.insert(
                i + j,
                f"{portfolios[0]}_{covariance[i]}_{expacted_return[j]}",
                p.msr(cov, er),
            )
    return all_msr


def all_gmv(cov_arr):
    all_gmv = pd.DataFrame()
    for i, cov in enumerate(cov_arr):
        print(f"{portfolios[1]}_{covariance[i]}")
        all_gmv.insert(i, f"{portfolios[1]}_{covariance[i]}", p.gmv(cov))
    return all_gmv


def weights_csv(index_names):
    for key in index_names:
        r = d.get_returns_df(d.icr_m[f"{key}"])
        cm = d.get_mkt_cap(d.icr_m[f"{key}"])
        cov = [o.sample_cov(r), o.cc_cov(r), o.shrinkage_cov(r)]
        er = [
            o.annualize_rets(r, 12),
            o.implied_returns(sigma=cov[0], w=p.weight_cw(cm), delta=2.5),
        ]
        all_msr(cov, er).to_csv(f"{key}_MSR")


weights_csv(d.index_names)
