import pickle

import sklearn.metrics as skm
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    explained_variance_score, mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, roc_auc_score
from ml.utils.help_functions import compute_diffs, plot_confusion_matrix



def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


with open("../../feature_selection/selected_81/selected_training.pkl", "rb") as f:
    X_train = pickle.load(f)
    truthClass_train = pickle.load(f)
    truthMean_train = pickle.load(f)

with open("../../feature_selection/selected_81/selected_big.pkl", "rb") as f:
    X_test = pickle.load(f)
    truthClass_test = pickle.load(f)
    truthMean_test = pickle.load(f)

clf = Ridge(alpha=0.1, random_state=42)
clf.fit(X_train, truthMean_train)
truthMean_pred = clf.predict(X_test)

nmse = normalized_mean_squared_error(truthMean_test, truthMean_pred)
mse = mean_squared_error(truthMean_test, truthMean_pred)
evs = explained_variance_score(truthMean_test, truthMean_pred)
r2 = r2_score(truthMean_test, truthMean_pred)
mae = mean_absolute_error(truthMean_test, truthMean_pred)
med_a_e = median_absolute_error(truthMean_test, truthMean_pred)

truthClass_test = [0 if t == 'no-clickbait' else 1 for t in truthClass_test]
truthClass_pred = [0 if t < 0.5 else 1 for t in truthMean_pred]


tn, fp, fn, tp = conf_matrix = confusion_matrix(truthClass_test, truthClass_pred).ravel()
print("(TN, FP, FN, TP) = {}".format((tn, fp, fn, tp)))

compute_diffs(truthClass_test, truthClass_pred, truthMean_test, truthMean_pred)
plot_confusion_matrix(truthClass_test, truthClass_pred, title='Ridge regression confusion matrix')


accuracy = accuracy_score(truthClass_test, truthClass_pred)
auc = roc_auc_score(truthClass_test, truthClass_pred)
precision = precision_score(truthClass_test, truthClass_pred)
recall = recall_score(truthClass_test, truthClass_pred)
f1 = f1_score(truthClass_test, truthClass_pred)

print('----------------Regression metrics-------------------------\n')
print("Explained Variance Score is ", evs)
print("Mean Squared Error  is ", mse)
print("Normalized Mean Squared Error  is ", nmse)
print("Mean Absolute Error  is ", mae)
print("Median Absolute Error  is ", med_a_e)
print("R2 score is ", r2)

print('\n----------------Classification metrics-------------------------\n')
print("Accuracy is ", accuracy)
print("AuC is", auc)
print("Precision is", precision)
print("Recall is", recall)
print("F1-core is ", f1)
