import pprint
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit

from feature_extraction.services.utils.classification_features_and_labels import get_features_and_labels

X, y = get_features_and_labels()
sss = StratifiedShuffleSplit(n_splits=10, random_state=42)

accuracy = 0
precision = 0
recall = 0
f1 = 0
f_scores = []

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # std_scale = StandardScaler().fit(X_train)
    # X_train = std_scale.transform(X_train)
    # X_test = std_scale.transform(X_test)

    clf = svm.SVC(kernel='rbf', C=198.85295716582226, gamma=0.059683872490462968, class_weight='balanced',
                  random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy += accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = conf_matrix = confusion_matrix(y_test, y_pred).ravel()
    print("(TN, FP, FN, TP) = {}".format((tn, fp, fn, tp)))

    precision += precision_score(y_test, y_pred)
    recall += recall_score(y_test, y_pred)
    f_scores.append(f1_score(y_test, y_pred))
    f1 += f1_score(y_test, y_pred)

print("Accuracy is ", accuracy / 10)

print("Precision is", precision / 10)

print("Recall is", recall / 10)

print("F1-core is ", f1 / 10)

f_scores = np.array(f_scores)
v = (np.var(f_scores) ** (0.5)) * 2

print("Variance is: {0}".format(v))
pprint.pprint(f_scores)
