from feature_extraction.services.image_service import get_text_from_image
from functools import reduce
import nltk
from nltk.corpus import stopwords


def calculate_all_behaviour_features(entry):
    behaviour_post_title = calculate_special_signs(entry["postText"])
    behaviour_post_image = calculate_special_signs(get_text_from_image(entry))
    behaviour_article_title = calculate_special_signs(entry["targetTitle"])
    behaviour_post_desc = calculate_special_signs(entry["targetDescription"])
    behaviour_post_keywords = calculate_special_signs(entry["targetKeywords"])
    behaviour_post_captions = calculate_special_signs(" ".join(entry["targetCaptions"]))
    behaviour_post_paragraphs = calculate_special_signs(" ".join(entry["targetParagraphs"]))

    return [behaviour_post_title, behaviour_post_image, behaviour_article_title, \
            behaviour_post_desc, behaviour_post_keywords, behaviour_post_captions, behaviour_post_paragraphs]


def get_feat_names():
    behaviour_post_title_feats = ['@_in_post_title', '!_in_post_title', '#_in_post_title', '?_in_post_title',
                                  'links_in_post_title', 'avg_word_len_post_title', 'fr_stop_words_post_title']
    behaviour_post_image_feats = ['@_in_post_image', '!_in_post_image', '#_in_post_image', '?_in_post_image',
                                  'links_in_post_image', 'avg_word_len_post_image', 'fr_stop_words_post_image']
    behaviour_article_title_feats = ['@_in_article_title', '!_in_article_title', '#_in_article_title',
                                     '?_in_article_title', 'links_in_article_title', 'avg_word_len_article_title',
                                     'fr_stop_words_article_title']
    behaviour_article_desc_feats = ['@_in_article_desc', '!_in_article_desc', '#_in_article_desc', '?_in_article_desc',
                                    'links_in_article_desc', 'avg_word_len_article_desc', 'fr_stop_words_article_desc']
    behaviour_article_keywords_feats = ['@_in_article_keywords', '!_in_article_keywords', '#_in_article_keywords',
                                        '?_in_article_keywords', 'links_in_article_keywords',
                                        'avg_word_len_article_keywords', 'fr_stop_words_article_keywords']
    behaviour_article_captions_feats = ['@_in_article_captions', '!_in_article_captions', '#_in_article_captions',
                                        '?_in_article_captions', 'links_in_article_captions',
                                        'avg_word_len_article_captions', 'fr_stop_words_article_captions']
    behaviour_article_paragraphs_feats = ['@_in_article_paragraphs', '!_in_article_paragraphs',
                                          '#_in_article_paragraphs', '?_in_article_paragraphs',
                                          'links_in_article_paragraphs', 'avg_word_len_article_paragraphs',
                                          'fr_stop_words_article_paragraphs']
    return behaviour_post_title_feats + behaviour_post_image_feats + behaviour_article_title_feats + \
           behaviour_article_desc_feats + behaviour_article_keywords_feats + behaviour_article_captions_feats + \
           behaviour_article_paragraphs_feats


def calculate_special_signs(text):
    # Calculate no of occurrences of @ ! # and ? and links
    tokenizer1 = nltk.RegexpTokenizer(r'https?://(?:[-\w./.]|(?:%[\da-fA-F]{2}))+')
    tokenizer2 = nltk.RegexpTokenizer(r'\w+')
    return text.count("@"), text.count("!"), text.count("#"), text.count("?"), len(tokenizer1.tokenize(text)), \
           0 if not list(map(lambda x: len(x), tokenizer2.tokenize(text))) else reduce((lambda x, y: x + y), list(
               map(lambda x: len(x), tokenizer2.tokenize(text)))) / len(tokenizer2.tokenize(text)), \
            len([w for w in tokenizer2.tokenize(text) if w.lower() in stopwords.words('english')])\
            / len(tokenizer2.tokenize(text))
