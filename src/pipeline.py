import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import ( drop_unnecessary_columns,handle_missing_values,encode_categorical_variables,feature_engineering)
from train_models import train_models
from evaluate_models import evaluate_models
from hyperparameter_tuning import tune_random_forest
from model_saving import save_model
from model_saving import load_model

def load_data(path):

    df = pd.read_csv(path)

    return df


def split_features_target(df):

    X = df.drop("Survived", axis=1)

    y = df["Survived"]

    return X, y

def create_train_test_split(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


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
    data = feature_engineering(data)
    data=encode_categorical_variables(data)

    # print(data.head())

    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    # print("Training set shape:", X_train.shape)
    # print("Testing set shape:", X_test.shape)
    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    # print(results)
    best_rf, best_params = tune_random_forest(X_train, y_train)

    # print("Best Random Forest Parameters:", best_params)

    save_model(best_rf, "models/best_model.pkl")

    loaded_model = load_model("models/best_model.pkl")

    sample = X_test.iloc[:5]

    predictions = loaded_model.predict(sample)

    print("Predictions:", predictions)