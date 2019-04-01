import nltk
from nltk import word_tokenize

with open("../feature_extraction/resources/slang_words.txt", encoding="utf-8") as f:
    content = f.readlines()
slang_list = [x.strip() for x in content]


def calculate_num_slang_words(text):
    if text == "":
        return 0
    words = word_tokenize(text)
    is_slang_list = [word in slang_list for word in words]
    return is_slang_list.count(True)


def calculate_all_num_slang_words(entry):
    slang_words_post_title = calculate_num_slang_words(entry["postText"][0])
    slang_words_article_title = calculate_num_slang_words(entry["targetTitle"])
    slang_words_post_desc = calculate_num_slang_words(entry["targetDescription"])
    slang_words_post_keywords = calculate_num_slang_words(entry["targetKeywords"])
    slang_words_post_captions = calculate_num_slang_words(" ".join(entry["targetCaptions"]))
    slang_words_post_paragraphs = calculate_num_slang_words(" ".join(entry["targetParagraphs"]))
    print([slang_words_post_title, slang_words_article_title, \
            slang_words_post_desc, slang_words_post_keywords, slang_words_post_captions, slang_words_post_paragraphs]
)
    return [slang_words_post_title, slang_words_article_title, \
            slang_words_post_desc, slang_words_post_keywords, slang_words_post_captions, slang_words_post_paragraphs]


def get_feat_names():
    return "post_text_slang_words", "article_title_slang_words", "article_desc_slang_words", \
           "article_keywords_slang_words", "article_captions_slang_words", \
           "article_paragraphs_slang_words"
