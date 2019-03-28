import json

import jsonpickle
import pandas as pd


def get_features_and_labels():

    filename = "../data_reading/data.txt"
    with open(filename) as json_file:
        data_encoded = json.load(json_file)
        data = jsonpickle.decode(data_encoded)

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
                clickbait = 1
            else:
                clickbait = 0
            post_dict[post["id"]]["label"] = clickbait

    features = pd.DataFrame.from_dict(post_dict, orient='index', dtype=None)

    labels_df = features.loc[:, 'label']
    features_df = features.drop('label', 1)

    X = features_df.values
    y = labels_df.values
    return X, y



