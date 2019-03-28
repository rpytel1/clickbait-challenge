from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from ml.svm.get_features_and_labels import get_features_and_labels

X, y = get_features_and_labels()
sss = StratifiedShuffleSplit(n_splits=10, random_state=42)


feat_num = len(X[0])
for i in range(5,12):
    accuracy = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    f_scores = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pca = PCA(n_components=i)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        clf = svm.SVC(kernel='rbf', gamma=0.1, class_weight='balanced')
        # clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy += accuracy_score(y_test, y_pred) * 100

        tn, fp, fn, tp = conf_matrix = confusion_matrix(y_test, y_pred).ravel()
        print("(TN, FP, FN, TP) = {}".format((tn, fp, fn, tp)))

        precision += precision_score(y_test, y_pred) * 100
        recall += recall_score(y_test, y_pred) * 100
        f_scores.append(f1_score(y_test, y_pred) * 100)
        f1 += f1_score(y_test, y_pred) * 100

    print("Accuracy is ", accuracy / 10)

    print("Precision is", precision / 10)

    print("Recall is", recall / 10)

    print("F1-core is ", f1 / 10)
