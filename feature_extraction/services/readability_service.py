import textstat


def get_readability(entry):
    readability_post_title = textstat.flesch_kincaid_grade(entry["postText"][0])
    readability_article_title = textstat.flesch_kincaid_grade(entry["targetTitle"])
    readability_post_desc = textstat.flesch_kincaid_grade(entry["targetDescription"])
    readability_post_keywords = textstat.flesch_kincaid_grade(entry["targetKeywords"])
    readability_post_captions = textstat.flesch_kincaid_grade(" ".join(entry["targetCaptions"]))
    readability_post_paragraphs = textstat.flesch_kincaid_grade(" ".join(entry["targetParagraphs"]))

    return [readability_post_title, readability_article_title, readability_post_desc, \
            readability_post_keywords, readability_post_captions, readability_post_paragraphs]


def get_feat_names():
    return "readability_post_text", "readability_article_title", "readability_post_desc", "readability_post_keywords", \
           "readability_post_captions", "readability_post_paragraphs"