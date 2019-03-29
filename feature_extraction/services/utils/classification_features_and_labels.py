import json

import pickle
import pandas as pd


def get_features_and_labels():

    filename = "../data_reading/data.obj"
    with open(filename, 'rb') as json_file:
        data = pickle.load(json_file)

    post_dict = {}
    for post in data:
        post_dict[post.id] = {}
        for ind, feature in enumerate(post.features):
            post_dict[post.id][ind] = feature

    truth_file = '../data/clickbait-training/truth.jsonl'
    with open(truth_file, encoding="utf-8") as f:
        for ind, line in enumerate(f):
            post = json.loads(line)
            if post["truthClass"] == "no-clickbait":
                clickbait = 0
            else:
                clickbait = 1
            post_dict[post["id"]]["label"] = clickbait

    features = pd.DataFrame.from_dict(post_dict, orient='index', dtype=None)

    labels_df = features.loc[:, 'label']
    features_df = features.drop('label', 1)

    X = features_df.values
    y = labels_df.values
    return X, y



