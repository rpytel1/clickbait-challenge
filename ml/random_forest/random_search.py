import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, StratifiedKFold
from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def rf_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)
    sss = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }

    n_iter_search = 10
    random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=random_grid,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=sss, refit='mse', verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X, y)
    print(random_search.best_params_)


if __name__ == "__main__":
    # X, _, truthMean = get_features_and_labels()
    #
    with open("../../feature_selection/selected_79/selected_with_pos.pkl", "rb") as f:
        X = pickle.load(f)
        truthClass = pickle.load(f)
        truthMean = pickle.load(f)

    rf_randomized_search(X, truthMean)
