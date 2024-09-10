import pandas as pd
import yfinance as yf

index_names = {
    "SP500": "^GSPC",
    "NasdaqComposite": "^IXIC",
    "DowJones": "^DJI",
    "FTSE100": "^FTSE",
    "DAX": "^GDAXI",
    "HSI": "^HSI",
    # Not added yet:
    # "^RUT": "Russell2000",
    # "^FCHI": "CAC40",
    # "^N225": "Nikkei225",
}

icr_d = {
    name: f"https://github.com/Bashlykov-Nikita/Companies-Returns/blob/main/{name}_d.csv?raw=true"
    for name in index_names.keys()
}


icr_m = {
    name: f"https://github.com/Bashlykov-Nikita/Companies-Returns/blob/main/{name}_m.csv?raw=true"
    for name in index_names.keys()
}


def get_returns_df(url):
    return pd.read_csv(url, index_col=0)


def get_mkt_cap(url):
    mkt_cap_df = []
    comp_names = get_returns_df(url).columns
    for company in comp_names:
        buff = yf.Ticker(company).info["marketCap"]
        mkt_cap_df.append(buff)

    return pd.DataFrame(mkt_cap_df, index=comp_names, columns=["Market Cap"])


def get_cap_weights(market_cap):
    total_mkt_cap = market_cap.sum(axis=0)
    cap_weight = market_cap.divide(total_mkt_cap)
    return cap_weight
