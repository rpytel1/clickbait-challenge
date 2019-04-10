import re, math
from collections import Counter

WORD = re.compile(r'\w+')

# function calculating cosine given two vectors
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

# function changing text into vector format
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

# aggregate funciton calculating cosine similiarity between all the parts of the article + post
def calculate_cosine_similiarity(entry):
    vector_post_title = text_to_vector(entry["postText"][0])
    vector_article_title = text_to_vector(entry["targetTitle"])
    vector_article_desc = text_to_vector(entry["targetDescription"])
    vector_article_keywords = text_to_vector(entry["targetKeywords"])
    vector_article_captions = text_to_vector(" ".join(entry["targetCaptions"]))
    vector_article_paragraph = text_to_vector(" ".join(entry["targetParagraphs"]))

    matrix_list = [vector_article_title, vector_article_desc, vector_article_keywords, vector_article_captions,
                   vector_article_paragraph]
    cosine_list = []
    for j in range(len(matrix_list)):
        cosine_list.append(get_cosine(vector_post_title, matrix_list[j]))
    return cosine_list


def get_feat_names():
    return ["cosine_post_article_title", "cosine_post_article_desc", "cosine_post_article_keywords",
            "cosine_post_captions", "cosine_post_paragraphs"]
