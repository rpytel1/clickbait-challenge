import nltk
import json
from data_reading.preprocess_data import apply_lower, contractions
from nltk import StanfordPOSTagger, StanfordNERTagger

model_en_tag = '../stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz'
jar_en_tag = '../stanford-ner-2018-10-16/stanford-ner-3.9.2.jar'
tagger2 = StanfordNERTagger(model_en_tag, path_to_jar=jar_en_tag, encoding='UTF-8')
model_pos_tag = '../stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
jar_pos_tag = '../stanford-postagger-2018-10-16/stanford-postagger.jar'
tagger_pos = StanfordPOSTagger(model_pos_tag, path_to_jar=jar_pos_tag, encoding='UTF-8')

def pattern_of_pos(entry):
    entry = contractions(entry["targetTitle"])
    entities = apply_lower(entry)

    # # apply lower to all words, not only to the non-entities words
    # apply_all_lower = []
    # for i in entities: apply_all_lower.append(i.lower())
    # entities = apply_all_lower

    # # Stanford POS tagging

    tagged_text = tagger_pos.tag(entities)


    tagged_text_en = tagger2.tag(entities)

    # # nltk POS tagging
    # tagged_text = nltk.pos_tag(entities)
    # pattern = 0
    # for i in range(len(tagged_text)-3):
    #
    #     # # checking if cardinal number exists
    #     # if tagged_text[i][1] == "CD"\
    #
    #     # # checking if digit exists
    #     if tagged_text[i][0].isdigit()\
    #     and ((tagged_text[i+1][1] == "NN" or tagged_text[i+1][1] == "NNS" or tagged_text[i+1][1] == "NNP" or tagged_text[i+1][1] == "NNPS")\
    #     or ((tagged_text[i+1][1] == "NN" or tagged_text[i+1][1] == "NNS" or tagged_text[i+1][1] == "NNP" or tagged_text[i+1][1] == "NNPS")\
    #     and (tagged_text[i+2][1] == "NN" or tagged_text[i+2][1] == "NNS" or tagged_text[i+2][1] == "NNP" or tagged_text[i+2][1] == "NNPS")))\
    #     and (((tagged_text[i+2][1] == "VB" or tagged_text[i+2][1] == "MD")\
    #     or (tagged_text[i+3][1] == "VB" or tagged_text[i+3][1] == "MD"))\
    #     or ((tagged_text[i+2][0] == "that") or (tagged_text[i+3][0] == "that"))):
    #         pattern = 1
    #
    # return pattern

    pattern = 0
    # a = 3
    # for i in range(a):
    if tagged_text[0][1] == 'CD' and tagged_text_en[0][1] != 'DATE':
        pattern = 1

    return pattern

#>>>>>>>>>> ['the', '10', 'hardest', 'oxbridge', 'degrees', 'to', 'get', 'accepted', 'on'] <<<<<<<<<<<<


def get_feat_names():
    return 'pattern exist'

# def read_data(filename):
#     data = []
#     with open(filename, encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data
#
#
# print('Dataset Reading...')
# data = read_data('../../data/clickbait-training/instances.jsonl')
# data_truth = read_data('../../data/clickbait-training/truth.jsonl')
# pattern = 0
# count = 0
# correct = 0
# clickbait_number = 0
# false_positive = 0
# for entry in data:
#     pattern = pattern_of_pos(entry)
#     print(data_truth[count]['truthClass'],"    ", pattern)
#     if data_truth[count]['truthClass'] == 'clickbait' and pattern == 1:
#         correct += 1
#         print("correct")
#     elif data_truth[count]['truthClass'] == 'non-clickbait' and pattern == 1:
#         false_positive += 1
#     if data_truth[count]['truthClass'] == 'clickbait':
#         clickbait_number += 1
#     count += 1
#
# print("number of clickbaits     ", clickbait_number)
# print("clickbait detected   ", correct)
# print("false_positive   ", false_positive)
