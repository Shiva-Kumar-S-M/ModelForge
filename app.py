import streamlit as st
import pandas as pd
import json

from src.model_saving import load_model
from src.data_preprocessing import (
    drop_unnecessary_columns,
    handle_missing_values,
    feature_engineering,
    encode_categorical_variables
)

# -----------------------------
# Load Model & Columns
# -----------------------------
model = load_model("models/best_model.pkl")
with open("models/train_columns.json", "r") as f:
    train_columns = json.load(f)


# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_input(df):

    df = drop_unnecessary_columns(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = encode_categorical_variables(df)

    # Align columns with training data
    df = df.reindex(columns=train_columns, fill_value=0)

    return df


# -----------------------------
# UI
# -----------------------------
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below:")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])


# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    # Preprocess
    processed_data = preprocess_input(input_data)

    # Predict
    prediction = model.predict(processed_data)[0]

    # Output
    if prediction == 1:
        st.success("🎉 Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")