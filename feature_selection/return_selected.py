from feature_extraction.services.utils.regression_features_and_labels import get_features_and_labels
import numpy as np
import json
import pickle


X, y_class, y_reg = get_features_and_labels()
with open('../data_reading/features_labels_big.json') as json_file:
    feat_names = json.load(json_file)

with open('selected_81/selected_features_labels_training.json') as json_file:
    feat_selected = json.load(json_file)

feat_dict = {}
for key, value in feat_names.items():
    if value not in feat_dict.keys():
        feat_dict[value] = key
print(len(feat_dict.keys()))

feat_dict_sel = {}
for key, value in feat_selected.items():
    if value not in feat_dict_sel.keys():
        feat_dict_sel[value] = key
print(len(feat_dict_sel.keys()))

selected = ()
for key in feat_dict_sel.keys():
    if key in feat_dict.keys():
        selected += (X[:, int(feat_dict[key])],)
final_X = np.column_stack(selected)
print(final_X.shape[0], final_X.shape[1])

f = open(r"selected_big.pkl", "wb")
pickle.dump(final_X, f)
pickle.dump(y_class, f)
pickle.dump(y_reg, f)
f.close()
