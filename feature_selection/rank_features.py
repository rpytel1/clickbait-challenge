from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels
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
    return SelectKBest(score_func=f_regression, k=X.shape[1]).fit_transform(X, y)


def get_kBest_mutual(X, y):
    return SelectKBest(score_func=mutual_info_regression, k="all").fit_transform(X, y)


def get_fpr_f_regr(X, y):
    return SelectFpr(score_func=f_regression).fit_transform(X, y)


def get_fpr_mutual(X, y):
    return SelectFpr(score_func=mutual_info_regression).fit_transform(X, y)


X, y_class, y_reg = get_features_and_labels()
with open('../data_reading/data_v1/features_labels.json') as json_file:
    feat_names = json.load(json_file)

# info gain ranking and values both for regression
feat_gain = get_info_gain_ranking(X, y_reg)
sort_feats = np.sort(np.array(feat_gain))[::-1]

print('Info gain for regression')
gain_ranking = []
for s in list(np.argsort(np.array(feat_gain))[::-1]):
    gain_ranking.append(feat_names[str(s)])
print('The features sorted by info gain: ', gain_ranking)
print('The feature values sorted by info gain: ', list(sort_feats))

# info gain ranking and values both for regression
k_best_gain_reg = get_kBest_f_regr(X, y_reg)
sort_feats_k_best_reg = np.sort(np.array(k_best_gain_reg))[::-1]

print('K-best for regression')
k_best_reg = []
for s in list(np.argsort(np.array(k_best_gain_reg))[::-1]):
    k_best_reg.append(feat_names[str(s)])
print('The features sorted by info gain: ', k_best_reg)
print('The feature values sorted by info gain: ', list(sort_feats_k_best_reg))
