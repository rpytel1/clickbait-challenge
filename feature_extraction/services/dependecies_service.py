import nltk
from data_reading.preprocess_data import apply_lower


def add_no_nouns(entry):
    entities = apply_lower(entry)

    # # apply lower to all words, not only to the non-entities words
    # apply_all_lower = []
    # for i in entities: apply_all_lower.append(i.lower())
    # entities = apply_all_lower

    tagged_text = nltk.pos_tag(entities)

    no_nouns = 0
    no_adverb = 0
    no_determiner = 0
    no_personal_pronoun = 0
    no_wh_determiner = 0
    no_possessive_pronoun = 0
    no_past_participle = 0  # VBN
    no_third_person = 0  # VBZ
    no_past_tense = 0  # VBD
    no_sing_present = 0  # VBP
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
    if len(tagged_text) != 0:
        no_nouns = no_nouns/len(tagged_text)
        no_adverb = no_adverb/len(tagged_text)
        no_determiner = no_determiner/len(tagged_text)
        no_wh_determiner = no_wh_determiner/len(tagged_text)
        no_personal_pronoun = no_personal_pronoun/len(tagged_text)
        no_possessive_pronoun = no_possessive_pronoun/len(tagged_text)
        no_past_participle = no_past_participle / len(tagged_text)
        no_third_person = no_third_person / len(tagged_text)
        no_past_tense = no_past_tense / len(tagged_text)
        no_sing_present = no_sing_present / len(tagged_text)

    return no_nouns, no_adverb, no_determiner, no_personal_pronoun, no_wh_determiner, no_possessive_pronoun,\
           no_past_participle, no_third_person, no_past_tense, no_past_tense, no_sing_present


def get_feat_names():
    return 'number of nouns', 'number of adverbs', 'number of determiners', 'number of personal pronouns', \
           'number of wh determiners', 'number of possessive pronouns', 'number of past participle verbs',\
           'number of third person verbs', 'number of past tense verbs', 'number of sing. present verbs'


