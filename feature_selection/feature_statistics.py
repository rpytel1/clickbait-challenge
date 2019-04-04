import json


def get_statistics(feat_names_grouped, feat_selected):
    selected_stat = {}
    category_length = {}
    for category in feat_names_grouped.keys():
        selected_stat[category] = 0
        category_length[category] = len(feat_names_grouped[category])
    for feature in feat_selected.values():
        for category in feat_names_grouped.keys():
            if feature in feat_names_grouped[category].values():
                selected_stat[category] += 1
                break
    for category in selected_stat.keys():
        selected_stat[category] /= category_length[category]
        selected_stat[category] = round(selected_stat[category], 2)
    return selected_stat


def find_indexes(feat_selected, feat_names_grouped):
    inds = []
    for feature in feat_selected.values():
        for category in feat_names_grouped.keys():
            if category != 'ngrams':
                for ind, name in feat_names_grouped[category].items():
                    if name == feature:
                        inds.append(ind)
    return inds

with open('../data_reading/feature_names_grouped.json') as json_file:
    feat_names_grouped = json.load(json_file)

with open('selected_78/selected_features_labels.json') as json_file:
    feat_selected = json.load(json_file)

print(get_statistics(feat_names_grouped, feat_selected))


