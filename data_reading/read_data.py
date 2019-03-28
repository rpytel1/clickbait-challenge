import json
import sys

from feature_extraction.extract_features import extract_features, save_models


def read_data(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


print('Dataset Reading...')
data = read_data('../data/clickbait-training/instances.jsonl')
print('Extracting Features...')
extract_features(data)
