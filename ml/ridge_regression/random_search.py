import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, StratifiedKFold
from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def rf_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)

    random_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}

    n_iter_search = 10
    random_search = RandomizedSearchCV(Ridge(), param_distributions=random_grid,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=3, refit='mse', verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X, y)
    print(random_search.best_params_)
    print(random_search.best_score_)


if __name__ == "__main__":
    with open("../../feature_selection/selected_81/selected_training.pkl", "rb") as f:
        X = pickle.load(f)
        truthClass = pickle.load(f)
        truthMean = pickle.load(f)
    rf_randomized_search(X, truthMean)
