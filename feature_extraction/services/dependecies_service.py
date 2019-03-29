import nltk
from data_reading.preprocess_data import apply_lower

def add_no_nouns(entry):
    entities = apply_lower(entry)

    # # apply lower to all words, not only to the non-entities words
    # apply_all_lower = []
    # for i in entities: apply_all_lower.append(i.lower())
    # entities = apply_all_lower

    tagged_text = nltk.pos_tag(entities)

    # print(tagged_text)
    no_nouns = 0
    no_adverb = 0
    no_determiner = 0
    no_personal_pronoun = 0
    no_wh_determiner = 0
    no_possessive_pronoun = 0
    for n in range(len(tagged_text)):
        if tagged_text[n][1] == "NN":
            no_nouns += 1
        if tagged_text[n][1] == "RB":
            no_adverb += 1
        if tagged_text[n][1] == "TD":
            no_determiner += 1
        if tagged_text[n][1] == "WTD":
            no_wh_determiner += 1
        if tagged_text[n][1] == "PRP":
            no_personal_pronoun += 1
        if tagged_text[n][1] == "PRP$":
            no_possessive_pronoun += 1
    if no_nouns != 0:
        no_nouns = no_nouns/len(tagged_text)
    if no_adverb != 0:
        no_adverb = no_adverb/len(tagged_text)
    if no_determiner != 0:
        no_determiner = no_determiner/len(tagged_text)
    if no_wh_determiner != 0:
        no_wh_determiner = no_wh_determiner/len(tagged_text)
    if no_personal_pronoun != 0:
        no_personal_pronoun = no_personal_pronoun/len(tagged_text)
    if no_possessive_pronoun != 0:
        no_possessive_pronoun = no_possessive_pronoun/len(tagged_text)

    return no_nouns, no_adverb, no_determiner, no_personal_pronoun, no_wh_determiner, no_possessive_pronoun


def get_feat_names():
    return 'number of nouns', 'number of adverbs', 'number of dertemriners', 'number of personal pronouns', \
           'number of wh determiners', 'number of possesive pronouns'

