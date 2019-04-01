import nltk
from data_reading.preprocess_data import apply_lower


def pattern_of_pos(entry):
    entities = apply_lower(entry)

    # # apply lower to all words, not only to the non-entities words
    # apply_all_lower = []
    # for i in entities: apply_all_lower.append(i.lower())
    # entities = apply_all_lower

    tagged_text = nltk.pos_tag(entities)
    pattern = 0
    for i in range(len(tagged_text)-3):
        # if tagged_text[i][1] == "CD"\
        if tagged_text[i][0].isdigit()\
        and ((tagged_text[i+1][1] == "NN" or tagged_text[i+1][1] == "NNS" or tagged_text[i+1][1] == "NNP" or tagged_text[i+1][1] == "NNPS")\
        or ((tagged_text[i+1][1] == "NN" or tagged_text[i+1][1] == "NNS" or tagged_text[i+1][1] == "NNP" or tagged_text[i+1][1] == "NNPS")\
        and (tagged_text[i+2][1] == "NN" or tagged_text[i+2][1] == "NNS" or tagged_text[i+2][1] == "NNP" or tagged_text[i+2][1] == "NNPS")))\
        and (((tagged_text[i+2][1] == "VB" or tagged_text[i+2][1] == "MD")\
        or (tagged_text[i+3][1] == "VB" or tagged_text[i+3][1] == "MD"))\
        or ((tagged_text[i+2][0] == "that") or (tagged_text[i+3][0] == "that"))):
            pattern = 1

    return pattern


def get_feat_names():
    return 'pattern exist'
