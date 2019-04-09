import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, StratifiedKFold
from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def rf_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)
    sss = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)

    random_grid = {'alpha': [0.1, 1.0, 10.0]}

    n_iter_search = 10
    random_search = RandomizedSearchCV(Ridge(), param_distributions=random_grid,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=sss, refit='mse', verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X, y)
    print(random_search.best_params_)


if __name__ == "__main__":
    # X, _, truthMean = get_features_and_labels()
    with open("../../feature_selection/selected_79/selected_with_pos.pkl", "rb") as f:
        X = pickle.load(f)
        truthClass = pickle.load(f)
        truthMean = pickle.load(f)
    rf_randomized_search(X, truthMean)
