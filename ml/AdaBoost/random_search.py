from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def rf_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)
    sss = StratifiedShuffleSplit(n_splits=10, random_state=42)

    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)

    # Create the random grid
    param_dist = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
        'loss': ['linear', 'square', 'exponential']
    }

    n_iter_search = 10
    random_search = RandomizedSearchCV(AdaBoostRegressor(), param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=sss, refit='mse', verbose=2, random_state=42, n_jobs=-1)

    random_search.fit(X, y)
    print(random_search.best_params_)


if __name__ == "__main__":
    X, _, truthMean = get_features_and_labels()
    rf_randomized_search(X, truthMean)
