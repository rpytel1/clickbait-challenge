from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st
from sklearn.ensemble import GradientBoostingRegressor
from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def gtboost_randomized_search(X, y):
    mse_scorer = make_scorer(mean_squared_error)

    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)

    # one_to_left = st.beta(10, 1)
    # from_zero_positive = st.expon(0, 50)
    #
    # params = {
    #     "n_estimators": st.randint(3, 40),
    #     "max_depth": st.randint(3, 40),
    #     "learning_rate": st.uniform(0.05, 0.4),
    #     "colsample_bytree": one_to_left,
    #     "subsample": one_to_left,
    #     "gamma": st.uniform(0, 10),
    #     'reg_alpha': from_zero_positive,
    #     "min_child_weight": from_zero_positive,
    # }
    #

    # Create the random grid
    params = {
        "n_estimators": st.randint(3, 40),
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "criterion": ["mse"]
    }

    n_iter_search = 10

    random_search = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=params,
                                       n_iter=n_iter_search,
                                       scoring={'mse': mse_scorer},
                                       cv=3, refit='mse', verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X, y)
    print(random_search.best_params_)


if __name__ == "__main__":
    X, _, truthMean = get_features_and_labels()
    gtboost_randomized_search(X, truthMean)
