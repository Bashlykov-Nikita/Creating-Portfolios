import pandas as pd
import numpy as np
import data as d
import options as o
import portfolios as p


covariance = ["Sample", "CCM", "Shrinage"]

expacted_return = ["Average", "Implied"]

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
            print(f"{portfolios[0]}_{covariance[i]}_{expacted_return[j]}")
            all_msr.insert(
                i + j,
                f"{portfolios[0]}_{covariance[i]}_{expacted_return[j]}",
                p.msr(cov, er),
            )
    all_msr.Name = "NasdaqComposite"
    return all_msr


all_msr(cov_arr, er_arr)


msr_test = pd.read_csv("SP500_MSR", index_col=0)
