import re
from collections import Counter
from statistics import mode

import numpy as np
from stanfordcorenlp import StanfordCoreNLP
import json

nlp = StanfordCoreNLP('http://localhost', port=9000,timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
props = {
            'annotators': 'sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
}


def annotate(sentence):
    return json.loads(modify_floating_point_sign(nlp.annotate(sentence, properties=props)))


def modify_floating_point_sign(str):
    return re.sub(r'(\d),(\d)', r'\1.\2', str)


def calculate_sentiment_features(str_list):
    sentiment_list = [2]
    for str in str_list:
        annotation = annotate(str)
        for s in annotation["sentences"]:
            sentiment_list.append(int(s["sentimentValue"]))

    matrix = np.array(sentiment_list)
    return get_mode(sentiment_list), matrix.mean()


def get_mode(lst):
    d_mem_count = Counter(lst)
    return d_mem_count.most_common()[0][0]


def calculate_all_sentiment_features(entry):
    post_title_sentiment = calculate_sentiment_features([entry["postText"][0]])
    article_title_sentiment = calculate_sentiment_features([entry["targetTitle"]])
    article_desc_sentiment = calculate_sentiment_features([entry["targetDescription"]])
    article_keywords_sentiment = calculate_sentiment_features([entry["targetKeywords"]])
    article_captions_sentiment = calculate_sentiment_features(entry["targetCaptions"])
    article_paragraphs_sentiment = calculate_sentiment_features(entry["targetParagraphs"])
    lst = [post_title_sentiment, article_title_sentiment, article_desc_sentiment, article_keywords_sentiment, article_captions_sentiment, article_paragraphs_sentiment]
    return list(sum(lst, ()))
