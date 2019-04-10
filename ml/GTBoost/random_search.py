from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from xgboost import XGBRegressor

from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def gtboost_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)
    sss = StratifiedShuffleSplit(n_splits=3, random_state=42)


    params = {
        "n_estimators": [50, 100, 200, 500, 1000],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "colsample_bytree": [0.3, 0.5, 0.8, 1],
        "subsample": [0.8, 0.9, 1],
        "gamma": [0, 1, 5]
    }


    n_iter_search = 100

    random_search = RandomizedSearchCV(XGBRegressor(), param_distributions=params,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=sss, refit='mse', verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X, y)
    print(random_search.best_params_)


if __name__ == "__main__":
    X, _, truthMean = get_features_and_labels()
    gtboost_randomized_search(X, truthMean)
