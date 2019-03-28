import sklearn.metrics as skm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    explained_variance_score, mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


X, truthClass, thruthMean = get_features_and_labels()
sss = StratifiedShuffleSplit(n_splits=10, random_state=42)

# initialize regression evaluation metrics
evs = 0
mse = 0
r2 = 0
mae = 0
med_a_e = 0
nmse = 0

# initialize classification evaluation metrics
accuracy = 0
precision = 0
recall = 0
f1 = 0
# f_scores = []

for train_index, test_index in sss.split(X, thruthMean):
    X_train, X_test = X[train_index], X[test_index]
    thruthMean_train, thruthMean_test = thruthMean[train_index], thruthMean[test_index]
    truthClass_train, truthClass_test = truthClass[train_index], truthClass[test_index]

    # std_scale = StandardScaler().fit(X_train)
    # X_train = std_scale.transform(X_train)
    # X_test = std_scale.transform(X_test)

    clf = RandomForestRegressor(max_features=X_train.shape[1], max_depth=None, criterion="mse", n_jobs=-1)
    clf.fit(X_train, thruthMean_train)
    thruthMean_pred = clf.predict(X_test)

    nmse += normalized_mean_squared_error(thruthMean_test, thruthMean_pred)
    mse += mean_squared_error(thruthMean_test, thruthMean_pred)
    evs += explained_variance_score(thruthMean_test, thruthMean_pred)
    r2 += r2_score(thruthMean_test, thruthMean_pred)
    mae += mean_absolute_error(thruthMean_test, thruthMean_pred)
    med_a_e += median_absolute_error(thruthMean_test, thruthMean_pred)

    truthClass_test = [0 if t == 'no-clickbait' else 1 for t in truthClass_test]
    truthClass_pred = [0 if t < 0.5 else 1 for t in thruthMean_pred]

    tn, fp, fn, tp = conf_matrix = confusion_matrix(truthClass_test, truthClass_pred).ravel()
    print("(TN, FP, FN, TP) = {}".format((tn, fp, fn, tp)))

    accuracy += accuracy_score(truthClass_test, truthClass_pred)
    precision += precision_score(truthClass_test, truthClass_pred)
    recall += recall_score(truthClass_test, truthClass_pred)
    f1 += f1_score(truthClass_test, truthClass_pred)

print('----------------Regression metrics-------------------------\n')
print("Explained Variance Score is ", evs / 10)
print("Mean Squared Error  is ", mse / 10)
print("Normalized Mean Squared Error  is ", nmse / 10)
print("Mean Absolute Error  is ", mae / 10)
print("Median Absolute Error  is ", med_a_e / 10)
print("R2 score is ", r2 / 10)

print('\n----------------Classification metrics-------------------------\n')
print("Accuracy is ", accuracy / 10)
print("Precision is", precision / 10)
print("Recall is", recall / 10)
print("F1-core is ", f1 / 10)
