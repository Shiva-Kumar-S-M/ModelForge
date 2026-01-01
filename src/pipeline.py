import pandas as pd


def load_data(path):

    df = pd.read_csv(path)

    return df


if __name__ == "__main__":

    data = load_data("data/raw/train.csv")

    print("First 5 rows:")
    print(data.head())

    print("\nDataset shape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns)