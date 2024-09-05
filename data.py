import pandas as pd

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
