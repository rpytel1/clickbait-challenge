import scipy.stats
from sklearn import svm
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def svr_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)
    sss = StratifiedShuffleSplit(n_splits=10, random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    kernels = ["rbf"]
    param_dist = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
                  'kernel': kernels}

    n_iter_search = 10
    random_search = RandomizedSearchCV(svm.SVR(), param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=sss, refit='mse', verbose=2, random_state=42, n_jobs=-1)

    random_search.fit(X, y)
    print(random_search.best_params_)


if __name__ == "__main__":
    X, _, truthMean = get_features_and_labels()
    svr_randomized_search(X, truthMean)
