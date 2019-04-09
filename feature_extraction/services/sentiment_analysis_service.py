import re
from collections import Counter

import nltk
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
import json

nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=500000)  # , quiet=False, logging_level=logging.DEBUG)
props = {
    'annotators': 'sentiment',
    'pipelineLanguage': 'en',
    'outputFormat': 'json'
}
tokenizer = nltk.RegexpTokenizer(r'\w+')


def annotate(sentence):
    return json.loads(modify_floating_point_sign(nlp.annotate(sentence, properties=props)))


def modify_floating_point_sign(str):
    return re.sub(r'(\d),(\d)', r'\1.\2', str)


def calculate_sentiment_features(str_list):
    if str_list == [''] or str_list == []:
        return 0, 0, 0, 0, 0
    sentiment_list = [2]
    num_extremes_positive = 0
    num_extremes_negative = 0
    syntactic_dist_list = [0]
    for str in str_list:
        annotation = annotate(str)
        index = 0
        for s in annotation["sentences"]:
            # Sentiment analysis
            sentiment_list.append(int(s["sentimentValue"]))

            # Maximum attribute
            lst = extract_sentiments_per_words(s)
            num_extremes_negative += lst.count('0')
            num_extremes_positive += lst.count('4')

            # Syntactic distance
            syntactic_dist_list.append(extract_length_of_syntactic_distance(s))
            index += 1
    matrix = np.array(sentiment_list)
    num_words = len(tokenizer.tokenize(" ".join(str_list)))
    if num_words == 0:
        num_words=1
    return get_mode(sentiment_list), matrix.mean(), num_extremes_positive / num_words, num_extremes_negative / num_words\
        , max(syntactic_dist_list)


def extract_sentiments_per_words(s):
    lst = re.findall("(sentiment=\d)+", s['sentimentTree'])
    improved_lst = [x[10:] for x in lst]
    return improved_lst


# Here we assume that both nlpcore and nltk tokenize sentences and worlds the same way
def extract_length_of_syntactic_distance(s):
    words = [token["word"] for token in s["tokens"]]
    dist_list = [0]
    for elem in s["basicDependencies"]:
        if elem["dep"] not in ['ROOT','punct']: #We take into account only into the words
            try:
                governor_gloss_id = words.index(elem["governorGloss"])
                dependent_gloss_id = words.index(elem["dependentGloss"])
                dist_list.append(abs(governor_gloss_id - dependent_gloss_id))
            except KeyError:
                dist_list.append(-1)
    return max(dist_list)


def get_mode(lst):
    d_mem_count = Counter(lst)
    return d_mem_count.most_common()[0][0]


# no keywords
def calculate_all_sentiment_features(entry):
    post_title_sentiment = calculate_sentiment_features([entry["postText"][0]])
    article_title_sentiment = calculate_sentiment_features([entry["targetTitle"]])
    article_desc_sentiment = calculate_sentiment_features([entry["targetDescription"]])
    article_keywords_sentiment = calculate_sentiment_features([entry["targetKeywords"]])
    article_paragraphs_sentiment = calculate_sentiment_features(entry["targetParagraphs"])
    lst = [post_title_sentiment, article_title_sentiment, article_desc_sentiment, article_keywords_sentiment,
            article_paragraphs_sentiment]
    return list(sum(lst, ()))


def get_feat_names():
    return ["avg_sentiment_post_text", "mode_sentiment_post_text",
            "num_of_positives_post_text", "num_of_negatives_post_text", "syntactic_dist_post_text",
            "avg_sentiment_article_title", "mode_sentiment_article_title",
            "num_of_positives_article_title", "num_of_negatives_article_title", "syntactic_dist_article_title",
            "avg_sentiment_article_description", "mode_sentiment_article_description",
            "num_of_positives_article_description", "num_of_negatives_article_description",
            "syntactic_dist_article_description",
            "avg_sentiment_article_keywords", "mode_sentiment_article_keywords",
            "num_of_positives_article_keywords", "num_of_negatives_article_keywords", "syntactic_dist_article_keywords",
            "avg_sentiment_article_paragraphs", "mode_sentiment_article_paragraphs",
            "num_of_positives_article_paragraphs", "num_of_negatives_article_paragraphs",
            "syntactic_dist_article_paragraphs"]
