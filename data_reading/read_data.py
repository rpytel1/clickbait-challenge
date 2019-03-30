import json

from data_reading import preprocess_data
from feature_extraction.extract_features import extract_features


def read_data(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


print('Dataset Reading...')
data = read_data('../data/clickbait-training/instances.jsonl')

# just preprocessing steps -- stop word removal, porter stemming, replacing numbers and urls --
print('Data preprocessing')
non_stop_word_data = preprocess_data.remove_stop_words(data)
stemmed_data = preprocess_data.apply_stemming(non_stop_word_data)
num_replaced_data = preprocess_data.replace_numbers(stemmed_data)
tagged_link_data = preprocess_data.find_links(num_replaced_data)
# no_link_data = preprocess_data.remove_links(num_replaced_data)

print('Extracting Features...')
extract_features(data)
