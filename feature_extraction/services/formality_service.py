from PyDictionary import PyDictionary
from nltk.corpus import words

from feature_extraction.services.common_words_service import get_unique_set_from_text
from feature_extraction.services.image_service import get_text_from_image

# Implementation of formality features
def calculate_all_formality_features(entry):
    formality_post_title = calculate_formal_words(entry["postText"][0])
    # formality_post_image = calculate_formal_words(get_text_from_image(entry))
    formality_article_title = calculate_formal_words(entry["targetTitle"])
    formality_post_desc = calculate_formal_words(entry["targetDescription"])
    formality_post_keywords = calculate_formal_words(entry["targetKeywords"])
    formality_post_captions = calculate_formal_words(" ".join(entry["targetCaptions"]))
    formality_post_paragraphs = calculate_formal_words(" ".join(entry["targetParagraphs"]))

    return formality_post_title, formality_article_title, \
           formality_post_desc, formality_post_keywords, formality_post_captions, formality_post_paragraphs


def calculate_formal_words(text):
    # TODO: rethink if we want unique set here
    text_set = get_unique_set_from_text(text)
    informal_num = 0
    for word in text_set:
        # try:
        #     PyDictionary.meaning(word)  # if not in dictionary it throws an exception, NOTE: swag is formal somehow..
        # except:
        #     informal_num += 1
        if word not in words.words():
            informal_num += 1
    formal_num = len(text_set) - informal_num
    return formal_num, informal_num, 0 if len(text_set) == 0 else formal_num / len(text_set), \
        0 if len(text_set) == 0 else informal_num / len(text_set)
