from data_reading.preprocess_data import apply_lower, contractions
from nltk import StanfordPOSTagger
import json

model_pos_tag = '../stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
jar_pos_tag = '../stanford-postagger-2018-10-16/stanford-postagger.jar'

tagger_pos = StanfordPOSTagger(model_pos_tag, path_to_jar=jar_pos_tag, encoding='UTF-8')

# function calculating different kinds of POS
def add_no_nouns(entry):
    entry = contractions(entry["targetTitle"])
    entities = apply_lower(entry)

    tagged_text = tagger_pos.tag(entities)

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
        if tagged_text[n][1] == "DT":
            no_determiner += 1
        if tagged_text[n][1] == "WDT":
            no_wh_determiner += 1
        if tagged_text[n][1] == "PRP":
            no_personal_pronoun += 1
        if tagged_text[n][1] == "PRP$":
            no_possessive_pronoun += 1
        if tagged_text[n][1] == "VBN":
            no_past_participle += 1
        if tagged_text[n][1] == "VBZ":
            no_third_person += 1
        if tagged_text[n][1] == "VBD":
            no_past_tense += 1
        if tagged_text[n][1] == "VBP":
            no_sing_present =+ 1
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
           no_past_participle, no_third_person, no_past_tense, no_sing_present


def get_feat_names():
    return 'number of nouns', 'number of adverbs', 'number of determiners', 'number of personal pronouns', \
           'number of wh determiners', 'number of possessive pronouns', 'number of past participle verbs',\
           'number of third person verbs', 'number of past tense verbs', 'number of sing. present verbs'

# function creating POS data
def create_POS_data(data):
    POS_list = []
    for entry in data:
        entry2 = contractions(entry["targetTitle"])
        entities = apply_lower(entry2)
        tagged_text = tagger_pos.tag(entities)
        for i in tagged_text:
            POS_list.append(i[1])
    return ' '.join(POS_list)

# function creating POS entry
def create_POS_entry(entry):
    POS_list = []
    entry = contractions(entry["targetTitle"])
    entities = apply_lower(entry)

    tagged_text = tagger_pos.tag(entities)
    for i in tagged_text:
        POS_list.append(i[1])
    return ' '.join(POS_list)
