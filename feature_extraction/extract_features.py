from model.model import Model


def extract_features(data):
    for entry in data:
        model = Model(id=entry["id"])
        add_length_features(model,entry)

def initialize_services():
    # TODO: Initialize tools like formality corpus

def add_length_features(model, entry):
    model.features.append([len(entry["postText"][0]), len(entry["targetTitle"]), len(entry["targetDescription"])])

def add_image_related_features(model, entry):
    # TODO: has multimedia + what text on multimedia

def add_diff_no_chars_features(model,entry):
    #TODO: add diffs

def add_character_ratio_features(model, entry):
    # TODO: add character ratio features

def add_no_words_features(model, entry):
    # TODO: no of words per each section

def add_words_ratio_features(model, entry):
    # TODO: add character ratio features

def add_commmon_words_features(model, entry):
    # TODO: no of words which overlap between set of keywords and words of certain part

def add_formality_features(model, entry):
    # TODO: calculate formal and informal words in each parts
    # TODO: calculate ratios

def add_time_features(model, entry):
    # TODO: calculate post creation hour?

def add_behaviour_analysis_features(model, entry):
    # TODO: check no of @ signs, no of hashtags

def add_article_properties_features(model, entry):
    # TODO: no of article keywords, no of paragraphs, no article captions