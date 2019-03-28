from ml.svm import get_features_and_labels
from info_gain import info_gain
import numpy as np
import json
from sklearn.feature_selection import SelectKBest, SelectFpr, f_regression, mutual_info_regression


def get_info_gain_ranking(X, y):
    feat_gain = []
    for j in range(X.shape[1]):
        feat_gain.append(info_gain.info_gain(X[:, j], y))
    return feat_gain


def get_kBest_f_regr(X, y):
    return SelectKBest(score_func=f_regression, k=X.shape[1]).fit(X, y)


def get_kBest_mutual(X, y):
    return SelectKBest(score_func=mutual_info_regression, k="all").fit(X, y)


def get_fpr_f_regr(X, y):
    return SelectFpr(score_func=f_regression).fit(X, y)


def get_fpr_mutual(X, y):
    return SelectFpr(score_func=mutual_info_regression).fit(X, y)


X, y = get_features_and_labels.get_features_and_labels()
with open('../data_reading/features_labels.json') as json_file:
    feat_names = json.load(json_file)

# info gain ranking and values
feat_gain = get_info_gain_ranking(X, y)
sort_feats = np.sort(np.array(feat_gain))[::-1]
gain_ranking = []
for s in list(np.argsort(np.array(feat_gain))[::-1]):
    gain_ranking.append(feat_names[str(s)])
print(list(sort_feats))
print(gain_ranking)

# TODO: check if the other selection methods work
