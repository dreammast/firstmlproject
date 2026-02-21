# Model hyperparameters configuration

model_params = {
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    "Decision Tree": {
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    "Linear Regression": {},
    "K-Neighbors Regressor": {
        "n_neighbors": 5
    },
    "XGBRegressor": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    },
    "CatBoosting Regressor": {
        "iterations": 100,
        "learning_rate": 0.1,
        "verbose": False
    },
    "AdaBoost Regressor": {
        "n_estimators": 100,
        "learning_rate": 0.1
    }
}
