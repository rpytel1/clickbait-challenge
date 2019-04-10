import pickle

import sklearn.metrics as skm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    explained_variance_score, mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, roc_auc_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels


def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


# X, truthClass, truthMean = get_features_and_labels()

with open("../../feature_selection/selected_81/selected_training.pkl", "rb") as f:
    X = pickle.load(f)
    truthClass = pickle.load(f)
    truthMean = pickle.load(f)

sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# initialize regression evaluation metrics
evs = 0
mse = 0
r2 = 0
mae = 0
med_a_e = 0
nmse = 0

# initialize classification evaluation metrics
accuracy = 0
auc = 0
precision = 0
recall = 0
f1 = 0

for train_index, test_index in sss.split(X, truthClass):
    X_train, X_test = X[train_index], X[test_index]
    truthMean_train, truthMean_test = truthMean[train_index], truthMean[test_index]
    truthClass_train, truthClass_test = truthClass[train_index], truthClass[test_index]

    std_scale = StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)

    clf = svm.SVR(kernel='linear', C=0.2, epsilon=0.2) #for 81 features
    clf = svm.SVR(kernel='linear', C=0.1, epsilon=0.1) #for all features
    # clf = svm.SVR(kernel='linear', C=0.015848931924611134)
    # clf = svm.SVR(kernel='linear', C=0.1024942976460832) without selection
    clf.fit(X_train, truthMean_train)
    truthMean_pred = clf.predict(X_test)

    nmse += normalized_mean_squared_error(truthMean_test, truthMean_pred)
    mse += mean_squared_error(truthMean_test, truthMean_pred)
    evs += explained_variance_score(truthMean_test, truthMean_pred)
    r2 += r2_score(truthMean_test, truthMean_pred)
    mae += mean_absolute_error(truthMean_test, truthMean_pred)
    med_a_e += median_absolute_error(truthMean_test, truthMean_pred)

    truthClass_test = [0 if t == 'no-clickbait' else 1 for t in truthClass_test]
    truthClass_pred = [0 if t < 0.5 else 1 for t in truthMean_pred]

    tn, fp, fn, tp = conf_matrix = confusion_matrix(truthClass_test, truthClass_pred).ravel()
    print("(TN, FP, FN, TP) = {}".format((tn, fp, fn, tp)))

    accuracy += accuracy_score(truthClass_test, truthClass_pred)
    auc += roc_auc_score(truthClass_test, truthClass_pred)
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
print("AuC is", auc / 10)
print("Precision is", precision / 10)
print("Recall is", recall / 10)
print("F1-core is ", f1 / 10)
