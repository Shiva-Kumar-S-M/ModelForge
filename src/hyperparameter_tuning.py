from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_random_forest(X_train, y_train):

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestClassifier()

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_params_