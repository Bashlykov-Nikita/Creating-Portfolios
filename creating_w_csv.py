import pandas as pd
import numpy as np
import data as d
import options as o
import portfolios as p


def weights_csv(index_names):
    for key in index_names:
        r = d.get_returns_df(d.icr_m[f"{key}"])
        er = [o.annualize_rets(r, 12), o.implied_returns()]
        cov = [o.sample_cov(r), o.cc_cov(r), o.shrinkage_cov(r)]
