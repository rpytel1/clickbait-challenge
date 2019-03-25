from nltk import RegexpTokenizer

from feature_extraction.services.image_service import get_text_from_image

tokenizer = RegexpTokenizer(r'\w+')


def calculate_common_words_features(entry):
    # no of words which overlap between set of keywords and words of certain part
    keywords = get_unique_set_from_text(entry["targetKeywords"])

    post_title = get_unique_set_from_text(entry["targetKeywords"])
    post_image = get_unique_set_from_text(get_text_from_image(entry))
    article_title = get_unique_set_from_text(entry["targetTitle"])
    article_desc = get_unique_set_from_text(entry["targetDescription"])
    article_captions = get_unique_set_from_text(" ".join(entry["targetCaptions"]))
    article_paragraphs = get_unique_set_from_text(" ".join(entry["targetParagraphs"]))

    return len(set.intersection(keywords, post_title)), len(set.intersection(keywords, post_image)), \
           len(set.intersection(keywords, article_title)), len(set.intersection(keywords, article_desc)), \
           len(set.intersection(keywords, article_captions)), len(set.intersection(keywords, article_paragraphs))


def get_unique_set_from_text(text):
    return set([x.lower() for x in tokenizer.tokenize(text)])
