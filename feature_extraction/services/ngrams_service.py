import string
import numpy as np
from nltk import ngrams, FreqDist, RegexpTokenizer
from nltk.corpus import stopwords

# function extracting unigrams from the entry
def extract_unigrams(text_ngrams):
    unigrams = list(ngrams(text_ngrams, 1))
    stop_words = set(stopwords.words('english'))
    unigrams = [w for w in unigrams if not w[0] in stop_words]
    unigrams_len = len(unigrams)
    unigrams_dist = list(FreqDist(unigrams).items())
    unigrams_freq = [(" ".join(w), f) for w, f in unigrams_dist]
    return dict(unigrams_freq)

# function extracting bigrams from the entry
def extract_bigrams(text_ngrams):
    bigrams = list(ngrams(text_ngrams, 2))
    bigrams_len = len(bigrams)
    bigrams_dist = list(FreqDist(bigrams).items())
    bigrams_freq = [(" ".join(w), f) for w, f in bigrams_dist]
    return dict(bigrams_freq)

# function extracting trigrams from the entry
def extract_trigrams(text_ngrams):
    trigrams = list(ngrams(text_ngrams, 3))
    trigrams_len = len(trigrams)
    trigrams_dist = list(FreqDist(trigrams).items())
    trigrams_freq = [(" ".join(w), f) for w, f in trigrams_dist]
    return dict(trigrams_freq)

# function extracting fourgrams from the entry
def extract_fourgrams(text_ngrams):
    fourgrams = list(ngrams(text_ngrams, 4))
    fourgrams_len = len(fourgrams)
    fourgrams_dist = list(FreqDist(fourgrams).items())
    fourgrams_freq = [(" ".join(w), f) for w, f in fourgrams_dist]
    return dict(fourgrams_freq)

# function extracting ngrams from the text
def extract_ngrams(text):
    # preprocess text
    text = text.lower().replace('\n', " ")
    text = text.translate(str.maketrans('', '', string.punctuation))

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

# aggregate function extracting all ngrams from each parts of the article+post
def calculate_all_ngrams(entry):
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

# function finding all possible ngrams
def find_final_ngrams(data):
    possible_ngrams = [{} for _ in range(20)]
    for doc_id, entry in enumerate(data):
        all_ngrams = calculate_all_ngrams(entry)
        for ind, ngram_dict in enumerate(all_ngrams):
            for ngram_name in ngram_dict:
                if str(ngram_name) not in possible_ngrams[ind]:
                    possible_ngrams[ind][str(ngram_name)] = [ngram_dict[ngram_name], 1, {doc_id: 1}]
                else:
                    possible_ngrams[ind][str(ngram_name)][0] += ngram_dict[ngram_name]
                    if doc_id not in possible_ngrams[ind][str(ngram_name)][2]:
                        possible_ngrams[ind][str(ngram_name)][2][doc_id] = 1
                        possible_ngrams[ind][str(ngram_name)][1] +=1
    # limits = [0.003, 0.002, 0.003, 0.003]
    print("Length of Data is {}".format(len(data)))
    df = np.floor(0.20*len(data))
    tf = np.round(0.40*len(data))
    print("DF={0}, TF={1}\n".format(df, tf))
    delete_keys = []
    for ind, category in enumerate(possible_ngrams):
        length = sum([v[0] for k, v in category.items()])
        for key in category.keys():
            if possible_ngrams[ind][str(key)][0] >= tf and possible_ngrams[ind][str(key)][1] >= df:
                possible_ngrams[ind][str(key)] = possible_ngrams[ind][str(key)][0] / length
            else:
                delete_keys += [(ind, str(key))]
    for tup in delete_keys:
        del possible_ngrams[tup[0]][tup[1]]
    return possible_ngrams

# function returning all possible ngrams
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
