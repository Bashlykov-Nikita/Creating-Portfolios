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


def get_returns_df(url: str) -> pd.DataFrame:
    return pd.read_csv(url, index_col=0)


def get_mkt_cap(url: str) -> pd.DataFrame:
    """Retrieve market capitalization data for companies listed in the provided URL.

    Args:
        url (str): URL to the CSV file containing company returns data.

    Returns:
        pandas.DataFrame: DataFrame with market capitalization values for each company.
    """
    mkt_cap_df = []
    comp_names = get_returns_df(url).columns
    for company in comp_names:
        buff = yf.Ticker(company).info["marketCap"]
        mkt_cap_df.append(buff)

    return pd.DataFrame(mkt_cap_df, index=comp_names)
