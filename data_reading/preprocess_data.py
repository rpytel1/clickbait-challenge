from nltk.corpus import stopwords
import nltk
from nltk import ne_chunk, pos_tag, Tree
from nltk.stem import PorterStemmer
import re
import html
from nltk import StanfordPOSTagger, StanfordNERTagger
from feature_extraction.resources import cList

model_pos_tag = '../stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
jar_pos_tag = '../stanford-postagger-2018-10-16/stanford-postagger.jar'

model_en_tag = '../stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'
jar_en_tag = '../stanford-ner-2018-10-16/stanford-ner-3.9.2.jar'

tagger_pos = StanfordPOSTagger(model_pos_tag, path_to_jar=jar_pos_tag, encoding='UTF-8')

tagger_en = StanfordNERTagger(model_en_tag, path_to_jar=jar_en_tag, encoding='UTF-8')

# preprocessing helper function to obtain string without html tags
def html_and_remove(entry):
    return re.sub(r'<.*?>', '', html.unescape(entry))

# aggregate function removing all html tags from data
def remove_html_tags(data):
    for count, entry in enumerate(data):
        print(count)
        entry['postText'][0] = html_and_remove(entry['postText'][0])
        entry['targetTitle'] = html_and_remove(entry['targetTitle'])
        entry['targetDescription'] = html_and_remove(entry['targetDescription'])
        entry['targetKeywords'] = html_and_remove(entry['targetKeywords'])
        for ind, par in enumerate(entry['targetParagraphs']):
            entry['targetParagraphs'][ind] = html_and_remove(entry['targetParagraphs'][ind])
        for ind, par in enumerate(entry['targetCaptions']):
            entry['targetCaptions'][ind] = html_and_remove(entry['targetCaptions'][ind])
    return data

# preprocessing helper function removing stopwords from string
def check_and_remove(tokenizer, entry, stopword_dict):
    words = tokenizer.tokenize(entry)
    for word in words:
        if word.lower() in stopword_dict.keys():
            entry = re.sub(r'\b'+word+r'\b', '', entry)
    return entry

# aggregate function removing all stop words from data
def remove_stop_words(data):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    stopword_dict = {w: 1 for w in stopwords.words('english')}
    for count, entry in enumerate(data):
        print(count)
        entry['postText'][0] = check_and_remove(tokenizer, entry['postText'][0], stopword_dict)
        entry['targetTitle'] = check_and_remove(tokenizer, entry['targetTitle'], stopword_dict)
        entry['targetDescription'] = check_and_remove(tokenizer, entry['targetDescription'], stopword_dict)
        entry['targetKeywords'] = check_and_remove(tokenizer, entry['targetKeywords'], stopword_dict)
        for ind, par in enumerate(entry['targetParagraphs']):
            entry['targetParagraphs'][ind] = check_and_remove(tokenizer, entry['targetParagraphs'][ind], stopword_dict)
        for ind, par in enumerate(entry['targetCaptions']):
            entry['targetCaptions'][ind] = check_and_remove(tokenizer, entry['targetCaptions'][ind], stopword_dict)
    return data

# preprocessing helper funciton removing numbers
def check_and_replace(tokenizer, entry):
    numbers = tokenizer.tokenize(entry)
    for num in numbers:
        entry = re.sub(r'\b' + num + r'\b', '[n]', entry)
    return entry

# aggregate function replacing all numbers in data to "num" keyword
def replace_numbers(data):
    tokenizer = nltk.RegexpTokenizer(r'\b[-+]?\d+\.\d+\b|\b\d+\b')
    for count, entry in enumerate(data):
        print(count)
        entry['postText'][0] = check_and_replace(tokenizer, entry['postText'][0])
        entry['targetTitle'] = check_and_replace(tokenizer, entry['targetTitle'])
        entry['targetDescription'] = check_and_replace(tokenizer, entry['targetDescription'])
        entry['targetKeywords'] = check_and_replace(tokenizer, entry['targetKeywords'])
        for ind, par in enumerate(entry['targetParagraphs']):
            entry['targetParagraphs'][ind] = check_and_replace(tokenizer, entry['targetParagraphs'][ind])
        for ind, par in enumerate(entry['targetCaptions']):
            entry['targetCaptions'][ind] = check_and_replace(tokenizer, entry['targetCaptions'][ind])
    return data

# preprocessing helper function removing numbers
def num_and_remove(tokenizer, entry):
    words = tokenizer.tokenize(entry)
    to_be_removed = [w for w in words if w.isalnum() and not w.isdigit() and not w.isalpha()]
    for w in to_be_removed:
        entry = entry.replace(w, re.sub(r'\d+', '', w))
    return entry

# aggregate function removing numbers from data
def remove_numbers(data):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for count, entry in enumerate(data):
        print(count)
        entry['postText'][0] = num_and_remove(tokenizer, entry['postText'][0])
        entry['targetTitle'] = num_and_remove(tokenizer, entry['targetTitle'])
        entry['targetDescription'] = num_and_remove(tokenizer, entry['targetDescription'])
        entry['targetKeywords'] = num_and_remove(tokenizer, entry['targetKeywords'])
        for ind, par in enumerate(entry['targetParagraphs']):
            entry['targetParagraphs'][ind] = num_and_remove(tokenizer, entry['targetParagraphs'][ind])
        for ind, par in enumerate(entry['targetCaptions']):
            entry['targetCaptions'][ind] = num_and_remove(tokenizer, entry['targetCaptions'][ind])
    return data

# helper function performing stemming
def stem_and_replace(tokenizer, stemmer, entry):
    words = tokenizer.tokenize(entry)
    for word in words:
        entry = entry.replace(word, stemmer.stem(word.lower()))
    return entry

# aggreagate function performing Porter stemming
def apply_stemming(data):
    tokenizer = nltk.RegexpTokenizer(r'\w+|[\[\w\]]+')
    ps = PorterStemmer()
    for count, entry in enumerate(data):
        print(count)
        entry['postText'][0] = stem_and_replace(tokenizer, ps, entry['postText'][0])
        entry['targetTitle'] = stem_and_replace(tokenizer, ps, entry['targetTitle'])
        entry['targetDescription'] = stem_and_replace(tokenizer, ps, entry['targetDescription'])
        entry['targetKeywords'] = stem_and_replace(tokenizer, ps, entry['targetKeywords'])
        for ind, par in enumerate(entry['targetParagraphs']):
            entry['targetParagraphs'][ind] = stem_and_replace(tokenizer, ps, entry['targetParagraphs'][ind])
        for ind, par in enumerate(entry['targetCaptions']):
            entry['targetCaptions'][ind] = stem_and_replace(tokenizer, ps, entry['targetCaptions'][ind])
    return data

# helper function replacing links with [url] keyword
def link_and_replace(tokenizer, entry):
    links = tokenizer.tokenize(entry)
    for link in links:
        entry = entry.replace(link, '[url]')
    return entry

# aggregate function replacing urls in all data
def find_links(data):
    tokenizer = nltk.RegexpTokenizer(r'https?://(?:[-\w./.]|(?:%[\da-fA-F]{2}))+')
    for count, entry in enumerate(data):
        print(count)
        entry['postText'][0] = link_and_replace(tokenizer, entry['postText'][0])
        entry['targetTitle'] = link_and_replace(tokenizer, entry['targetTitle'])
        entry['targetDescription'] = link_and_replace(tokenizer, entry['targetDescription'])
        entry['targetKeywords'] = link_and_replace(tokenizer, entry['targetKeywords'])
        for ind, par in enumerate(entry['targetParagraphs']):
            entry['targetParagraphs'][ind] = link_and_replace(tokenizer, entry['targetParagraphs'][ind])
        for ind, par in enumerate(entry['targetCaptions']):
            entry['targetCaptions'][ind] = link_and_replace(tokenizer, entry['targetCaptions'][ind])
    return data

# helper function removing links
def link_and_remove(tokenizer, entry):
    links = tokenizer.tokenize(entry)
    for link in links:
        entry = entry.replace(link, '')
    return entry

# aggregate function removing all links from the data
def remove_links(data):
    tokenizer = nltk.RegexpTokenizer(r'https?://(?:[-\w./.]|(?:%[\da-fA-F]{2}))+')
    for count, entry in enumerate(data):
        print(count)
        entry['postText'][0] = link_and_remove(tokenizer, entry['postText'][0])
        entry['targetTitle'] = link_and_remove(tokenizer, entry['targetTitle'])
        entry['targetDescription'] = link_and_remove(tokenizer, entry['targetDescription'])
        entry['targetKeywords'] = link_and_remove(tokenizer, entry['targetKeywords'])
        for ind, par in enumerate(entry['targetParagraphs']):
            entry['targetParagraphs'][ind] = link_and_remove(tokenizer, entry['targetParagraphs'][ind])
        for ind, par in enumerate(entry['targetCaptions']):
            entry['targetCaptions'][ind] = link_and_remove(tokenizer, entry['targetCaptions'][ind])
    return data


def apply_lower(data):
    text = data
    tokenizer = nltk.RegexpTokenizer('\w+')
    entities = tokenizer.tokenize(text)

    tagged_text_pos = tagger_pos.tag(entities)
    tagged_text_en = tagger_en.tag(entities)
    entities_en = []
    j = 0

    for i in tagged_text_en:
        if i[1] != "O":
            entities_en.append(tagged_text_pos[j][0])
        else:
            entities_en.append(tagged_text_pos[j][0].lower())
        j += 1

    return entities_en

def contractions(entry):
    expand = cList.expandContractions(entry)
    return expand


