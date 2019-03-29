import json
import pickle

from feature_extraction.services import image_service, common_words_service, time_service, behaviour_analysis_service, \
    cosine_similiarity_service, article_service, clickbait_words_service, dependecies_service, \
    sentiment_analysis_service
from feature_extraction.services.formality_service import calculate_all_formality_features
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
            add_cosine_similarities(model, entry)
            add_sentiment_features(model, entry)
            add_clickbait_phrases_check(model, entry)
            add_no_of_nouns(model, entry)

        else:
            feat_names.extend(add_image_related_features(model, entry))
            feat_names.extend(add_linguistic_analysis_features(model, entry))
            feat_names.extend(add_common_words_features(model, entry))
            feat_names.append(add_time_features(model, entry))
            feat_names.extend(add_behaviour_analysis_features(model, entry))
            feat_names.extend(add_article_properties_features(model, entry))
            feat_names.extend(add_cosine_similarities(model, entry))
            feat_names.extend(add_sentiment_features(model, entry))
            feat_names.append(add_clickbait_phrases_check(model, entry))
            feat_names.extend(add_no_of_nouns(model ,entry))
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
    model.features.extend(image_service.calculate_image_features(entry))
    return image_service.get_feat_names()


def add_linguistic_analysis_features(model, entry):
    # num, diffs and ratios for chars and words
    word_service = WordService()
    feat_list = list(word_service.calculate_all_linguistic_features(entry))
    model.features.extend(feat_list[:int(len(feat_list) / 2)])
    return feat_list[int(len(feat_list) / 2):]


def add_common_words_features(model, entry):
    model.features.extend(common_words_service.calculate_common_words_features(entry))
    return common_words_service.get_feat_names()


def add_formality_features(model, entry):
    model.features.extend(calculate_all_formality_features(entry))


def add_time_features(model, entry):
    # calculate post creation hour?
    model.features.append(time_service.calculate_time_features(entry))
    return time_service.get_feat_names()


def add_behaviour_analysis_features(model, entry):
    # check no of @ signs, no of hashtags
    model.features.extend(behaviour_analysis_service.calculate_all_behaviour_features(entry))
    return behaviour_analysis_service.get_feat_names()


def add_article_properties_features(model, entry):
    # no of article keywords, no of paragraphs, no article captions
    model.features.extend(article_service.calculate_article_features(entry))
    return article_service.get_feat_names()


def add_no_of_nouns(model, entry):
    model.features.extend(dependecies_service.add_no_nouns(entry))
    # print(dependecies_service.add_no_nouns(entry))
    return dependecies_service.get_feat_names()

def add_cosine_similarities(model, entry):
    model.features.extend(cosine_similiarity_service.calculate_cosine_similiarity(entry))
    return cosine_similiarity_service.get_feat_names()


def add_sentiment_features(model, entry):
    model.features.extend(sentiment_analysis_service.calculate_all_sentiment_features(entry))
    return sentiment_analysis_service.get_feat_names()


def add_clickbait_phrases_check(model, entry):
    model.features.append(clickbait_words_service.get_clickbait_words_features(entry))
    print(model.features)
    return clickbait_words_service.get_feat_names()


def save_models(model_lists):
    with open('data.obj', 'wb') as f:
        pickle.dump(model_lists, f)
