#function calculating number of characters in keywords, number of keywords, number of paragraphs and number of captions
def calculate_article_features(entry):
    return 0 if not len(entry["targetKeywords"]) else len(entry["targetKeywords"].split(',')), \
           len(entry["targetParagraphs"]), len(entry["targetCaptions"])


def get_feat_names():
    return 'number_of_keywords', 'number_of_paragraphs', 'number_of_captions'
