import json
import sys

from feature_extraction.services.word_service import WordService


def read_data(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
#data = read_data(sys.argv[0])

print("aaaa")
text = "All work and no play makes jack dull boy, ? ! All work and no play makes jack a dull boy."
print(WordService.calculate_basic_linguistic_features(text))
