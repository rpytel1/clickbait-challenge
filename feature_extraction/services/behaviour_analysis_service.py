from feature_extraction.services.image_service import get_text_from_image


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
    behaviour_post_title_feats = ['@_in_post_title', '!_in_post_title', '#_in_post_title', '?_in_post_title']
    behaviour_post_image_feats = ['@_in_post_image', '!_in_post_image', '#_in_post_image', '?_in_post_image']
    behaviour_article_title_feats = ['@_in_article_title', '!_in_article_title', '#_in_article_title',
                                     '?_in_article_title']
    behaviour_article_desc_feats = ['@_in_article_desc', '!_in_article_desc', '#_in_article_desc', '?_in_article_desc']
    behaviour_article_keywords_feats = ['@_in_article_keywords', '!_in_article_keywords', '#_in_article_keywords',
                                        '?_in_article_keywords']
    behaviour_article_captions_feats = ['@_in_article_captions', '!_in_article_captions', '#_in_article_captions',
                                        '?_in_article_captions']
    behaviour_article_paragraphs_feats = ['@_in_article_paragraphs', '!_in_article_paragraphs',
                                          '#_in_article_paragraphs', '?_in_article_paragraphs']
    return behaviour_post_title_feats + behaviour_post_image_feats + behaviour_article_title_feats + \
           behaviour_article_desc_feats + behaviour_article_keywords_feats + behaviour_article_captions_feats + \
           behaviour_article_paragraphs_feats


def calculate_special_signs(text):
    # Calculate no of occurnces of @ ! # and ?
    return text.count("@"), text.count("!"), text.count("#"), text.count("?")
