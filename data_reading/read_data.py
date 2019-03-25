import json
import sys

from feature_extraction.extract_features import extract_features, save_models


def read_data(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
data = read_data(sys.argv[1])
extract_features(data)
