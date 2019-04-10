from data_reading.preprocess_data import apply_lower, contractions
from nltk import StanfordPOSTagger, StanfordNERTagger

model_en_tag = '../stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz'
jar_en_tag = '../stanford-ner-2018-10-16/stanford-ner-3.9.2.jar'
tagger2 = StanfordNERTagger(model_en_tag, path_to_jar=jar_en_tag, encoding='UTF-8')
model_pos_tag = '../stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
jar_pos_tag = '../stanford-postagger-2018-10-16/stanford-postagger.jar'
tagger_pos = StanfordPOSTagger(model_pos_tag, path_to_jar=jar_pos_tag, encoding='UTF-8')

#method to extract POS patterns
def pattern_of_pos(entry):
    entry = contractions(entry["targetTitle"])
    entities = apply_lower(entry)

    tagged_text = tagger_pos.tag(entities)


    tagged_text_en = tagger2.tag(entities)

    pattern = 0
    if tagged_text[0][1] == 'CD' and tagged_text_en[0][1] != 'DATE':
        pattern = 1

    return pattern

def get_feat_names():
    return 'pattern exist'
