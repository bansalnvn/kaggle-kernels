import pandas as pd


def read_input_data():
    _csv_file_name = "./data/veg_and_fruits.csv"
    df = pd.read_csv(_csv_file_name)
    df = df[~pd.isnull(df['price'])]
    df = df[df['price'] > 0.0]
    print(df)
    return df
