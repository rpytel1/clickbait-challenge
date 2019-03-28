def calculate_article_features(entry):
    return len(entry["targetKeywords"]), len(entry["targetParagraphs"]), len(entry["targetCaptions"]), \
           'number_of_keywords', 'number_of_paragraphs', 'number_of_captions'
