import json

import jsonpickle as jsonpickle

from feature_extraction.services.article_service import calculate_article_features
from feature_extraction.services.behaviour_analysis_service import calculate_all_behaviour_features
from feature_extraction.services.common_words_service import calculate_common_words_features
from feature_extraction.services.cosine_similiarity_service import calculate_cosine_similiarity
from feature_extraction.services.formality_service import calculate_all_formality_features
from feature_extraction.services.image_service import calculate_image_features
from feature_extraction.services.sentiment_analysis_service import calculate_all_sentiment_features
from feature_extraction.services.time_service import calculate_time_features
from feature_extraction.services.word_service import WordService
from model.model import Model


def extract_features(data):
    model_lists = []
    feat_names = []
    i = 0
    for entry in data:
        print(i)
        model = Model(index=entry["id"])
        if i != 0:
            add_image_related_features(model, entry)
            add_linguistic_analysis_features(model, entry)
            add_common_words_features(model, entry)
            # TODO: Formality check takes ages!!
            # add_formality_features(model,entry)
            add_time_features(model, entry)
            add_behaviour_analysis_features(model, entry)
            add_article_properties_features(model, entry)
        else:
            feat_names.extend(add_image_related_features(model, entry))
            feat_names.extend(add_linguistic_analysis_features(model, entry))
            feat_names.extend(add_common_words_features(model, entry))
            feat_names.extend(add_time_features(model, entry))
            feat_names.extend(add_behaviour_analysis_features(model, entry))
            feat_names.extend(add_article_properties_features(model, entry))
            print(len(feat_names))
            feat_dict = {}
            for j in range(len(feat_names)):
                feat_dict[j] = feat_names[j]
            with open('features_labels.json', 'w') as fp:
                json.dump(feat_dict, fp)
        model_lists.append(model)
        i += 1
    save_models(model_lists)


def add_image_related_features(model, entry):
    # has multimedia + what text on multimedia
    feat_list = list(calculate_image_features(entry))
    model.features.extend(feat_list[:int(len(feat_list)/2)])
    return feat_list[int(len(feat_list)/2):]


def add_linguistic_analysis_features(model, entry):
    # num, diffs and ratios for chars and words
    word_service = WordService()
    feat_list = list(word_service.calculate_all_linguistic_features(entry))
    model.features.extend(feat_list[:int(len(feat_list)/2)])
    return feat_list[int(len(feat_list)/2):]


def add_common_words_features(model, entry):
    feat_list = list(calculate_common_words_features(entry))
    model.features.extend(feat_list[:int(len(feat_list)/2)])
    return feat_list[int(len(feat_list)/2):]


def add_formality_features(model, entry):
    model.features.extend(calculate_all_formality_features(entry))


def add_time_features(model, entry):
    # calculate post creation hour?
    feat_list = list(calculate_time_features(entry))
    model.features.extend(feat_list[:int(len(feat_list)/2)])
    return feat_list[int(len(feat_list)/2):]


def add_behaviour_analysis_features(model, entry):
    # check no of @ signs, no of hashtags
    feat_list = list(calculate_all_behaviour_features(entry))
    model.features.extend(feat_list[:int(len(feat_list)/2)])
    return feat_list[int(len(feat_list)/2):]


def add_article_properties_features(model, entry):
    # no of article keywords, no of paragraphs, no article captions
    feat_list = list(calculate_article_features(entry))
    model.features.extend(feat_list[:int(len(feat_list)/2)])
    return feat_list[int(len(feat_list)/2):]



def add_cosine_similarities(model,entry):
    model.features.extend(calculate_cosine_similiarity(entry))


def add_sentiment_features(model,entry):
    model.features.extend(calculate_all_sentiment_features(entry))


def save_models(model_lists):
    with open('data.txt', 'w') as f:
        encoded_dictionary = jsonpickle.encode(model_lists)
        json.dump(encoded_dictionary, f)