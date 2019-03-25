from feature_extraction.services.article_service import calculate_article_features
from feature_extraction.services.behaviour_analysis_service import calculate_all_behaviour_features
from feature_extraction.services.common_words_service import calculate_common_words_features
from feature_extraction.services.formality_service import calculate_all_formality_features
from feature_extraction.services.image_service import calculate_image_features
from feature_extraction.services.time_service import calculate_time_features
from feature_extraction.services.word_service import WordService
from model.model import Model


def extract_features(data):
    for entry in data:
        model = Model(id=entry["id"])
        add_image_related_features(model, entry)
        add_linguistic_analysis_features(model, entry)
        add_common_words_features(model, entry)


def add_image_related_features(model, entry):
    # has multimedia + what text on multimedia
    model.features.extend(calculate_image_features(entry))


def add_linguistic_analysis_features(model, entry):
    # num, diffs and ratios for chars and words
    word_service = WordService()
    model.features.extend(word_service.calculate_all_linguistic_features(entry))


def add_common_words_features(model, entry):
    model.extend(calculate_common_words_features(entry))


def add_formality_features(model, entry):
    model.extend(calculate_all_formality_features(entry))


def add_time_features(model, entry):
    # TODO: calculate post creation hour?
    model.extend(calculate_time_features(entry))


def add_behaviour_analysis_features(model, entry):
    # TODO: check no of @ signs, no of hashtags
    model.extend(calculate_all_behaviour_features(entry))


def add_article_properties_features(model, entry):
    # TODO: no of article keywords, no of paragraphs, no article captions
    model.extend(calculate_article_features(entry))
