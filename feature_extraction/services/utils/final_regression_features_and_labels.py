import json
import pickle

import jsonpickle
import pandas as pd


def get_features_and_labels(case):

    if case == 'train':
        filename = '../../data_reading/data.obj'
    else:
        filename = '../../data_reading/data_big.obj'

    with open(filename, 'rb') as json_file:
        data = pickle.load(json_file)

    post_dict = {}
    for post in data:
        post_dict[post.id] = {}
        for ind, feature in enumerate(post.features):
            post_dict[post.id][ind] = feature

    if case == 'train':
        truth_file = '../../data/clickbait-training/truth.jsonl'
    else:
        truth_file = '../../data/clickbait17-validation-170630/truth.jsonl'

    with open(truth_file, encoding="utf-8") as f:
        for ind, line in enumerate(f):
            post = json.loads(line)
            if post["id"] in post_dict.keys():
                post_dict[post["id"]]["truthMean"] = post["truthMean"]
                post_dict[post["id"]]["truthClass"] = post["truthClass"]
            else:
                print('Didn\'t find ', post["id"])


    index = post_dict.keys()
    features = pd.DataFrame.from_dict(post_dict, orient='index', dtype=None)

    truthClass_df = features.loc[:, 'truthClass']
    truthMean_df = features.loc[:, 'truthMean']
    features_df = features.drop(['truthMean', 'truthClass'], axis=1)

    X = features_df.values
    truthClass = truthClass_df.values
    truthMean = truthMean_df.values
    return X, truthClass, truthMean



