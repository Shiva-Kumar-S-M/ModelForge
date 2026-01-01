import pandas as pd
from data_preprocessing import drop_unnecessary_columns,handle_missing_values


def load_data(path):

    df = pd.read_csv(path)

    return df


if __name__ == "__main__":

    data = load_data("data/raw/train.csv")

    # print("First 5 rows:")
    # print(data.head())

    # print("\nDataset shape:")
    # print(data.shape)

    # print("\nColumns:")
    # print(data.columns)

    data=drop_unnecessary_columns(data)

    data=handle_missing_values(data)

    print(data.head())