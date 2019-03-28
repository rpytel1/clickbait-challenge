import scipy
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from feature_extraction.services.utils.classification_features_and_labels import get_features_and_labels


def svm_randomized_search(X, y):
    f1_scorer = make_scorer(f1_score, average='micro')
    precision_scorer = make_scorer(precision_score, average='micro')
    recall_scorer = make_scorer(recall_score, average='micro')

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    kernels = ["rbf", "linear"]
    class_weight_range = [{0: 1, 1: 1}, 'balanced']
    param_dist = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
                  'kernel': kernels, 'class_weight': class_weight_range,
                  'decision_function_shape': ['ovo', 'ovr']}

    n_iter_search = 100
    random_search = RandomizedSearchCV(svm.SVC(), param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       scoring={'precision': precision_scorer, 'recall': recall_scorer,
                                                'f1': f1_scorer},
                                       cv=3, refit='f1', n_jobs=-1, random_state=42)

    random_search.fit(X, y)

    print(random_search.best_params_)


if __name__ == "__main__":
    X, y = get_features_and_labels()
    svm_randomized_search(X, y)
