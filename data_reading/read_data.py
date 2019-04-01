import json
import pickle

from data_reading import preprocess_data
from feature_extraction.extract_features import extract_features


def read_data(filename):
    data = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# print('Dataset Reading...')
# data = read_data('../data/clickbait-training/instances.jsonl')
#
# # just preprocessing steps -- stop word removal, porter stemming, replacing numbers and urls --
# print('Data preprocessing...')
#
# print('Stopword Removal')
# non_stop_word_data = preprocess_data.remove_stop_words(data)
#
# print('Replacing numbers with [n] and links with [url]')
# num_link_replaced_data = preprocess_data.replace_numbers(preprocess_data.find_links(data))
#
# print('Preparing data for ngrams')
# ngram_data = preprocess_data.apply_stemming(num_link_replaced_data)
#
# print('Stemming')
# removed_link_data = preprocess_data.remove_links(data)
# stemmed_no_link_data = preprocess_data.apply_stemming(removed_link_data)
#
# print('Removing links and numbers from alphanumerical words')
# num_link_removed_data = preprocess_data.remove_numbers(removed_link_data)
#
# print('Stopword Removal -> Removing links -> Stemming -> Replacing')
# all_in_data = preprocess_data.replace_numbers(preprocess_data.apply_stemming(preprocess_data.remove_links(non_stop_word_data)))
#
# f = open(r"preprocessed.pkl", "wb")
# pickle.dump(data, f)
# pickle.dump(num_link_replaced_data, f)
# pickle.dump(stemmed_no_link_data, f)
# pickle.dump(num_link_removed_data, f)
# pickle.dump(all_in_data, f)
# pickle.dump(ngram_data, f)
# pickle.dump(removed_link_data, f)
# f.close()

print('Opening preprocessed data...')
f = open("preprocessed.pkl", "rb")
data = pickle.load(f)
num_link_replaced_data = pickle.load(f)
stemmed_no_link_data = pickle.load(f)
num_link_removed_data = pickle.load(f)
all_in_data = pickle.load(f)
ngram_data = pickle.load(f)
removed_link_data = pickle.load(f)
f.close()

print('Extracting Features...')
extract_features(data, num_link_replaced_data, stemmed_no_link_data,
                 num_link_removed_data, all_in_data, ngram_data, removed_link_data)
