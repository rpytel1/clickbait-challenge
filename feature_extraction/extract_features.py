import json
import pickle

from feature_extraction.services import image_service, common_words_service, time_service, behaviour_analysis_service, \
cosine_similiarity_service, article_service, clickbait_words_service, dependecies_service, patternPOS_service, \
slang_service, readability_service, ngrams_service, sentiment_analysis_service

from feature_extraction.services.formality_service import calculate_all_formality_features
from feature_extraction.services.ngrams_service import find_final_ngrams
from feature_extraction.services.word_service import WordService
from model.model import Model


def extract_features(data, num_link_replaced_data, stemmed_no_link_data,
                     num_link_removed_data, all_in_data, ngram_data, removed_link_data):
    model_lists = []
    feat_names = []
    i = 0
    final_ngrams = find_final_ngrams(data)

    for raw, replaced, stemmed, removed, all_in, ngramish, no_link in \
            zip(data, num_link_replaced_data, stemmed_no_link_data, num_link_removed_data, all_in_data, ngram_data, removed_link_data):
        print(i)
        if not raw["id"] == replaced["id"] == stemmed["id"] == removed["id"] == all_in["id"] == ngramish["id"] == no_link["id"]:
            print('-------------- Malakia --------------')
            break
        model = Model(index=raw["id"])
        if i != 0:
            add_image_related_features(model, raw)
            add_linguistic_analysis_features(model, replaced)
            add_common_words_features(model, stemmed)
            add_time_features(model, raw)
            add_behaviour_analysis_features(model, raw)
            add_article_properties_features(model, raw)
            add_cosine_similarities(model, all_in)
            add_sentiment_features(model, replaced)
            add_clickbait_phrases_check(model, raw)
            add_no_of_pos_tagging(model, removed)
            add_pattern_pos(model, removed)
            add_slang_features(model, raw)
            add_readability_features(model, no_link)
            add_ngrams(model, ngramish, final_ngrams)
        else:
            feat_names.extend(add_image_related_features(model, raw))
            feat_names.extend(add_linguistic_analysis_features(model, replaced))
            feat_names.extend(add_common_words_features(model, stemmed))
            feat_names.extend(add_time_features(model, raw))
            feat_names.extend(add_behaviour_analysis_features(model, raw))
            feat_names.extend(add_article_properties_features(model, raw))
            feat_names.extend(add_cosine_similarities(model, all_in))
            feat_names.extend(add_sentiment_features(model, replaced))
            feat_names.extend(add_clickbait_phrases_check(model, raw))
            feat_names.extend(add_slang_features(model, removed))
            feat_names.extend(add_no_of_pos_tagging(model, removed))
            feat_names.append(add_pattern_pos(model, raw))
            feat_names.extend(add_readability_features(model, no_link))
            add_ngrams(model, ngramish, final_ngrams)
            feat_names.extend([k for i in final_ngrams for k in i])

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
    model.features.extend(word_service.calculate_all_linguistic_features(entry))
    return word_service.get_feat_names()


def add_common_words_features(model, entry):
    model.features.extend(common_words_service.calculate_common_words_features(entry))
    return common_words_service.get_feat_names()


def add_formality_features(model, entry):
    model.features.extend(calculate_all_formality_features(entry))


def add_time_features(model, entry):
    # calculate post creation hour?
    model.features.extend(time_service.calculate_time_features(entry))
    return time_service.get_feat_names()


def add_behaviour_analysis_features(model, entry):
    # check no of @ signs, no of hashtags
    model.features.extend(behaviour_analysis_service.calculate_all_behaviour_features(entry))
    return behaviour_analysis_service.get_feat_names()


def add_article_properties_features(model, entry):
    # no of article keywords, no of paragraphs, no article captions
    model.features.extend(article_service.calculate_article_features(entry))
    return article_service.get_feat_names()


def add_no_of_pos_tagging(model, entry):
    model.features.extend(dependecies_service.add_no_nouns(entry))
    # print(dependecies_service.add_no_nouns(entry))
    return dependecies_service.get_feat_names()


def add_pattern_pos(model,entry):
    model.features.append(patternPOS_service.pattern_of_pos(entry))
    return patternPOS_service.get_feat_names()


def add_cosine_similarities(model, entry):
    model.features.extend(cosine_similiarity_service.calculate_cosine_similiarity(entry))
    return cosine_similiarity_service.get_feat_names()


def add_sentiment_features(model, entry):
    model.features.extend(sentiment_analysis_service.calculate_all_sentiment_features(entry))
    return sentiment_analysis_service.get_feat_names()


def add_clickbait_phrases_check(model, entry):
    model.features.extend(clickbait_words_service.get_clickbait_words_features(entry))
    return clickbait_words_service.get_feat_names()


def add_slang_features(model, entry):
    model.features.extend(slang_service.calculate_all_num_slang_words(entry))
    return slang_service.get_feat_names()


def add_readability_features(model, entry):
    model.features.extend(readability_service.get_readability(entry))
    return readability_service.get_feat_names()


def add_ngrams(model, entry, final_ngrams):
    model.features.extend(ngrams_service.get_all_ngrams(entry, final_ngrams))


def save_models(model_lists):
    with open('data.obj', 'wb') as f:
        pickle.dump(model_lists, f)
