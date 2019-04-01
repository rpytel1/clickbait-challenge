import re
import string
import numpy as np
from nltk import ngrams, FreqDist, RegexpTokenizer
from nltk.corpus import stopwords


def extract_unigrams(text_ngrams):
    unigrams = list(ngrams(text_ngrams, 1))
    stop_words = set(stopwords.words('english'))
    unigrams = [w for w in unigrams if not w[0] in stop_words]
    unigrams_len = len(unigrams)
    unigrams_dist = list(FreqDist(unigrams).items())
    unigrams_freq = [(" ".join(w), f / unigrams_len) for w, f in unigrams_dist]
    return dict(unigrams_freq)


def extract_bigrams(text_ngrams):
    bigrams = list(ngrams(text_ngrams, 2))
    bigrams_len = len(bigrams)
    bigrams_dist = list(FreqDist(bigrams).items())
    bigrams_freq = [(w, f / bigrams_len) for w, f in bigrams_dist]
    return dict(bigrams_freq)


def extract_trigrams(text_ngrams):
    trigrams = list(ngrams(text_ngrams, 3))
    trigrams_len = len(trigrams)
    trigrams_dist = list(FreqDist(trigrams).items())
    trigrams_freq = [(w, f / trigrams_len) for w, f in trigrams_dist]
    return dict(trigrams_freq)


def extract_fourgrams(text_ngrams):
    fourgrams = list(ngrams(text_ngrams, 4))
    fourgrams_len = len(fourgrams)
    fourgrams_dist = list(FreqDist(fourgrams).items())
    fourgrams_freq = [(w, f / fourgrams_len) for w, f in fourgrams_dist]
    return dict(fourgrams_freq)


def extract_ngrams(text):
    # preprocess text
    text = text.lower().replace('\n', " ")
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text = re.sub('[^a-zA-Z0-9 \n\.]', '[_]', text)
    # text = re.sub('[-+]?\d*\.\d+|\d+', '[n]', text)

    # word tokens
    tokenizer = RegexpTokenizer(r'\w+|[\[\w\]]+')
    text_ngrams = tokenizer.tokenize(text)

    # unigrams
    unigrams = extract_unigrams(text_ngrams)

    # bigrams
    bigrams = extract_bigrams(text_ngrams)

    # trigrams
    trigrams = extract_trigrams(text_ngrams)

    # fourgrams
    fourgrams = extract_fourgrams(text_ngrams)

    return unigrams, bigrams, trigrams, fourgrams


def calculate_all_ngrams(entry):
    # targetKeywords_unigrams, targetKeywords_bigrams, targetKeywords_trigrams, targetKeywords_fourgrams = extract_ngrams(
    #     entry["targetKeywords"])
    # target_keywords = (entry["targetKeywords"]).split(',')
    postText_unigrams, postText_bigrams, postText_trigrams, postText_fourgrams = extract_ngrams(
        entry["postText"][0])
    targetTitle_unigrams, targetTitle_bigrams, targetTitle_trigrams, targetTitle_fourgrams = extract_ngrams(
        entry["targetTitle"])
    targetDescription_unigrams, targetDescription_bigrams, targetDescription_trigrams, targetDescription_fourgrams = extract_ngrams(
        entry["targetDescription"])
    # targetCaptions_unigrams, targetCaptions_bigrams, targetCaptions_trigrams, targetCaptions_fourgrams = extract_ngrams(
    #     " ".join(entry["targetCaptions"]))
    targetParagraphs_unigrams, targetParagraphs_bigrams, targetParagraphs_trigrams, targetParagraphs_fourgrams = extract_ngrams(
        " ".join(entry["targetParagraphs"]))

    return postText_unigrams, postText_bigrams, postText_trigrams, postText_fourgrams, \
           targetTitle_unigrams, targetTitle_bigrams, targetTitle_trigrams, targetTitle_fourgrams, \
           targetDescription_unigrams, targetDescription_bigrams, targetDescription_trigrams, targetDescription_fourgrams, \
           targetParagraphs_unigrams, targetParagraphs_bigrams, targetParagraphs_trigrams, targetParagraphs_fourgrams


def find_final_ngrams(data):
    # data = read_data('../../data/clickbait-training/instances.jsonl')
    possible_ngrams = [{} for _ in range(16)]
    for entry in data:
        all_ngrams = calculate_all_ngrams(entry)
        for ind, ngram_dict in enumerate(all_ngrams):
            for ngram_name in ngram_dict:
                if ngram_name not in possible_ngrams[ind]:
                    possible_ngrams[ind][ngram_name] = 1
                else:
                    possible_ngrams[ind][ngram_name] += 1
    limits = [0.0005, 0.0001, 0.0001, 0.0001]
    for ind, category in enumerate(possible_ngrams):
        length = len(category)
        temp = list(category.items())
        temp.sort(key=lambda tup: tup[1], reverse=True)
        possible_ngrams[ind] = dict(temp[:int(np.floor(limits[ind%4]*length))])
        # length = sum([v for k, v in category.items()])
        # for key in list(category):
        #     if possible_ngrams[ind][key] / length >= limits[ind%4]:
        #         possible_ngrams[ind][key] /= length
        #     else:
        #         del possible_ngrams[ind][key]
    final_ngrams = possible_ngrams

    return final_ngrams


def get_all_ngrams(entry, final_ngrams):
    final_features = []
    all_ngrams = list(calculate_all_ngrams(entry))
    for i, f in enumerate(all_ngrams):
        all_ngrams[i] = dict([(str(k), v) for k, v in f.items()])
    for ind, f in enumerate(all_ngrams):
        for k in final_ngrams[ind]:
            try:
                final_features += [f[k]]
            except KeyError:
                final_features += [0]
    return map(float, final_features)
