def calculate_article_features(entry):
    return len(entry["targetKeywords"]), len(entry["targetParagraphs"]), len(entry["targetCaptions"])
