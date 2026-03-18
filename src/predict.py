import pandas as pd
from model_saving import load_model


def preprocess_input(df):
    """
    Apply same preprocessing steps used during training
    """
    import json

    from data_preprocessing import (
        drop_unnecessary_columns,
        handle_missing_values,
        feature_engineering,
        encode_categorical_variables
    )

    df = drop_unnecessary_columns(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = encode_categorical_variables(df)

    with open("models/train_columns.json", "r") as f:
        train_columns = json.load(f)

    # Align columns
    df = df.reindex(columns=train_columns, fill_value=0)

    return df


def predict(input_data):
    """
    Load model and make predictions
    """

    model = load_model("models/best_model.pkl")

    processed_data = preprocess_input(input_data)

    predictions = model.predict(processed_data)

    return predictions


if __name__ == "__main__":

    sample = pd.DataFrame({
        "Pclass": [3],
        "Sex": ["male"],
        "Age": [22],
        "SibSp": [1],
        "Parch": [0],
        "Fare": [7.25],
        "Embarked": ["S"]
    })

    result = predict(sample)

    print("Prediction (0 = No, 1 = Yes):", result)