from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels
from info_gain import info_gain
import numpy as np
import json
import pickle
from sklearn.feature_selection import SelectKBest, SelectFpr, f_regression, mutual_info_regression
import statistics


def get_info_gain_ranking(X, y):
    feat_gain = []
    for j in range(X.shape[1]):
        feat_gain.append(info_gain.info_gain(y, X[:, j]))
    return feat_gain


def get_kBest_f_regr(X, y):
    return SelectKBest(score_func=f_regression, k=X.shape[1]).fit(X, y)


def get_kBest_mutual(X, y):
    return SelectKBest(score_func=mutual_info_regression, k="all").fit(X, y)


def get_fpr_f_regr(X, y):
    return SelectFpr(score_func=f_regression).fit(X, y)


def get_fpr_mutual(X, y):
    return SelectFpr(score_func=mutual_info_regression).fit(X, y)


X, y_class, y_reg = get_features_and_labels()
with open('../data_reading/features_labels.json') as json_file:
    feat_names = json.load(json_file)

# k-best ranking and values for regression
k_best_gain_reg = get_kBest_f_regr(X, y_reg)
sort_feats_k_best_reg = np.sort(np.nan_to_num(k_best_gain_reg.scores_))[::-1]

print('K-best for regression')
k_best_reg = []
for s in list(np.argsort(np.nan_to_num(k_best_gain_reg.scores_))[::-1]):
    k_best_reg.append(feat_names[str(s)])
print('The features sorted by k-best regression score: ', k_best_reg)
print('The feature values sorted by k-best regression score: ', list(sort_feats_k_best_reg))

# k-best ranking and values for mutual information
k_best_gain_mutual = get_kBest_mutual(X, y_reg)
sort_feats_k_best_mutual = np.sort(np.nan_to_num(k_best_gain_mutual.scores_))[::-1]

print('K-best for mutual information')
k_best_mutual = []
for s in list(np.argsort(np.nan_to_num(k_best_gain_mutual.scores_))[::-1]):
    k_best_mutual.append(feat_names[str(s)])
print('The features sorted by k-best mutual info score: ', k_best_mutual)
print('The feature values sorted by k-best mutual info score: ', list(sort_feats_k_best_mutual))

# fpr ranking and values for regression
fpr_gain_reg = get_fpr_f_regr(X, y_reg)
sort_feats_fpr_reg = np.sort(np.nan_to_num(fpr_gain_reg.scores_))[::-1]

print('FPR for regression')
fpr_reg = []
for s in list(np.argsort(np.nan_to_num(fpr_gain_reg.scores_))[::-1]):
    fpr_reg.append(feat_names[str(s)])
print('The features sorted by fpr regression score: ', fpr_reg)
print('The feature values sorted by fpr regression score: ', list(sort_feats_fpr_reg))

# fpr ranking and values for mutual info
fpr_gain_mutual = get_fpr_mutual(X, y_reg)
sort_feats_fpr_mutual = np.sort(np.nan_to_num(fpr_gain_mutual.scores_))[::-1]

print('FPR for mutual info')
fpr_mutual = []
for s in list(np.argsort(np.nan_to_num(fpr_gain_mutual.scores_))[::-1]):
    fpr_mutual.append(feat_names[str(s)])
print('The features sorted by fpr mutual info score: ', fpr_mutual)
print('The feature values sorted by fpr mutual info score: ', list(sort_feats_fpr_mutual))

voting = {}
for i in range(len(feat_names)):
    if k_best_reg[i] not in voting.keys():
        voting[k_best_reg[i]] = i
    else:
        voting[k_best_reg[i]] += i
    if k_best_mutual[i] not in voting.keys():
        voting[k_best_mutual[i]] = i
    else:
        voting[k_best_mutual[i]] += i
    if fpr_reg[i] not in voting.keys():
        voting[fpr_reg[i]] = i
    else:
        voting[fpr_reg[i]] += i
    if fpr_mutual[i] not in voting.keys():
        voting[fpr_mutual[i]] = i
    else:
        voting[fpr_mutual[i]] += i

feat_dict = {}
for key, value in feat_names.items():
    if value not in feat_dict.keys():
        feat_dict[value] = key
print(len(feat_dict.keys()))

selected = ()
selected_feat_dict = {}
for ind, feature in enumerate([x for x in list(voting.items()) if x[1] <= statistics.stdev(voting.values())]):
    selected += (X[:, int(feat_dict[feature[0]])],)
    selected_feat_dict[ind] = feature[0]
final_X = np.column_stack(selected)
print(final_X.shape[0], final_X.shape[1])

f = open(r"selected_with_pos.pkl", "wb")
pickle.dump(final_X, f)
pickle.dump(y_class, f)
pickle.dump(y_reg, f)
f.close()

with open('selected_features_labels_with_pos.json', 'w') as fp:
    json.dump(selected_feat_dict, fp)

# # info gain ranking and values
# feat_gain = get_info_gain_ranking(final_X, y_reg)
# sort_feats = np.sort(np.array(feat_gain))[::-1]
#
# print('Info gain for regression')
# gain_ranking = []
# for s in list(np.argsort(np.array(feat_gain))[::-1]):
#     gain_ranking.append(feat_names[str(s)])
# print('The features sorted by info gain: ', gain_ranking)
# print('The feature values sorted by info gain: ', list(sort_feats))
