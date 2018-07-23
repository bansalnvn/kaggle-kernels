import pandas as pd
from pathlib import Path

_DATA_STORE = "./data/pd.pickle"


def read_input_data():
    # check if the _DATA_STORE file exist.
    my_file = Path(_DATA_STORE)
    if my_file.is_file():
        df = pd.read_pickle(_DATA_STORE)
    else:
        today = pd.datetime.today()
        _csv_file_name = "./data/veg_and_fruits.csv"
        df = pd.read_csv(_csv_file_name)
        df = df[~pd.isnull(df['price'])]
        df = df[df['price'] > 0.0]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].between('1980-01-01', today)]
        df = df.filter(items=['Item Name', 'Date', 'price'])
        pd.to_pickle(df, _DATA_STORE)
    all_items = df['Item Name'].unique()
    return df, all_items
