import json
import sys


def read_data(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
data = read_data(sys.argv[0])

