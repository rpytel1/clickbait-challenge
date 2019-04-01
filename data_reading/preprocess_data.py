from nltk.corpus import stopwords
import nltk
from nltk import ne_chunk, pos_tag, Tree
from nltk.stem import PorterStemmer
import re


def check_and_remove(tokenizer, entry, stopword_dict):
    words = tokenizer.tokenize(entry)
    for word in words:
        if word.lower() in stopword_dict.keys():
            entry = entry.replace(word, '')
    return entry


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


def check_and_replace(tokenizer, entry):
    numbers = tokenizer.tokenize(entry)
    for num in numbers:
        entry = entry.replace(num, '[n]')
    return entry


def replace_numbers(data):
    tokenizer = nltk.RegexpTokenizer(r'[-+]?\d*\.\d+|\d+')
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


def num_and_remove(tokenizer, entry):
    words = tokenizer.tokenize(entry)
    to_be_removed = [w for w in words if w.isalnum() and not w.isdigit() and not w.isalpha()]
    for w in to_be_removed:
        entry = entry.replace(w, re.sub(r'\d+', '', w))
    return entry


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


def stem_and_replace(tokenizer, stemmer, entry):
    words = tokenizer.tokenize(entry)
    for word in words:
        entry = entry.replace(word, stemmer.stem(word.lower()))
    return entry


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


def link_and_replace(tokenizer, entry):
    links = tokenizer.tokenize(entry)
    for link in links:
        entry = entry.replace(link, '[url]')
    return entry


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


def link_and_remove(tokenizer, entry):
    links = tokenizer.tokenize(entry)
    for link in links:
        entry = entry.replace(link, '')
    return entry


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
    text = data["targetTitle"]
    tokenizer = nltk.RegexpTokenizer('\w+')
    entities = tokenizer.tokenize(text)
    tagged_text = pos_tag(entities)
    tagged_text_chunk = ne_chunk(tagged_text)
    entities = []
    for i in tagged_text_chunk:
        if type(i) == Tree:
            for j in range(len(i)):
                entities.append(i[j][0])
        else:
            entities.append(i[0].lower())
    return entities
