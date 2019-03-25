from feature_extraction.services.image_service import get_text_from_image


def calculate_all_behaviour_features(entry):
    behaviour_post_title = calculate_special_signs(entry["postText"])
    behaviour_post_image = calculate_special_signs(get_text_from_image(entry))
    behaviour_article_title = calculate_special_signs(entry["targetTitle"])
    behaviour_post_desc = calculate_special_signs(entry["targetDescription"])
    behaviour_post_keywords = calculate_special_signs(entry["targetKeywords"])
    behaviour_post_captions = calculate_special_signs(" ".join(entry["targetCaptions"]))
    behaviour_post_paragraphs = calculate_special_signs(" ".join(entry["targetParagraphs"]))

    return behaviour_post_title, behaviour_post_image, behaviour_article_title, \
           behaviour_post_desc, behaviour_post_keywords, behaviour_post_captions, behaviour_post_paragraphs


def calculate_special_signs(text):
    # Calculate no of occurnces of @ ! # and ?
    return text.count("@"), text.count("!"), text.count("#"), text.count("?")
