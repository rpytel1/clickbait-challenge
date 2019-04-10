import pickle

from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit


def rf_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)

    # Create the random grid
    param_dist = {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
        'loss': ['linear']
    }

    n_iter_search = 20
    random_search = RandomizedSearchCV(AdaBoostRegressor(), param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=3, refit='mse', verbose=2, random_state=42, n_jobs=4)

    random_search.fit(X, y)
    print(random_search.best_params_)


if __name__ == "__main__":
    with open("../../feature_selection/selected_81/selected_training.pkl", "rb") as f:
        X = pickle.load(f)
        truthClass = pickle.load(f)
        truthMean = pickle.load(f)

    rf_randomized_search(X, truthMean)
