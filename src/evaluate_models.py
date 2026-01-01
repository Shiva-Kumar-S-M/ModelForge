import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": accuracy
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by="Accuracy", ascending=False)

    return results_df