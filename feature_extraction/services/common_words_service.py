from nltk import RegexpTokenizer
from feature_extraction.services.image_service import get_text_from_image

tokenizer = RegexpTokenizer(r'\w+')

# function calculating no of words which overlap between set of keywords and words of certain part
def calculate_common_words_features(entry):
    keywords = get_unique_set_from_text(entry["targetKeywords"])

    post_title = get_unique_set_from_text(entry["postText"][0])
    post_image = get_unique_set_from_text(get_text_from_image(entry))
    article_title = get_unique_set_from_text(entry["targetTitle"])
    article_desc = get_unique_set_from_text(entry["targetDescription"])
    article_captions = get_unique_set_from_text(" ".join(entry["targetCaptions"]))
    article_paragraphs = get_unique_set_from_text(" ".join(entry["targetParagraphs"]))

    return len(set.intersection(keywords, post_title)), len(set.intersection(keywords, post_image)), \
           len(set.intersection(keywords, article_title)), len(set.intersection(keywords, article_desc)), \
           len(set.intersection(keywords, article_captions)), len(set.intersection(keywords, article_paragraphs))


def get_feat_names():
    return 'keywords_in_post_title', 'keywords_in_post_image', 'keywords_in_article_title', 'keywords_in_article_desc', \
           'keywords_in_article_captions', 'keywords_in_article_paragraphs'


def get_unique_set_from_text(text):
    return set([x.lower() for x in tokenizer.tokenize(text)])
