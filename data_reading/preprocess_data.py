from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer


def check_and_remove(tokenizer, entry, name):
    words = tokenizer.tokenize(entry[name])
    for word in words:
        if word.lower() in stopwords.words('english'):
            entry[name] = entry[name].replace(word, '')
    return entry


def remove_stop_words(data):
    filtered_data = []
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for count, entry in enumerate(data):
        print(count)
        words = tokenizer.tokenize(entry['postText'][0])
        for word in words:
            if word.lower() in stopwords.words('english'):
                entry['postText'][0] = entry['postText'][0].replace(word, '')
        entry = check_and_remove(tokenizer, entry, 'targetTitle')
        entry = check_and_remove(tokenizer, entry, 'targetDescription')
        entry = check_and_remove(tokenizer, entry, 'targetKeywords')
        for ind, par in enumerate(entry['targetParagraphs']):
            words = tokenizer.tokenize(par)
            for word in words:
                if word.lower() in stopwords.words('english'):
                    entry['targetParagraphs'][ind] = entry['targetParagraphs'][ind].replace(word, '')
        for ind, par in enumerate(entry['targetCaptions']):
            words = tokenizer.tokenize(par)
            for word in words:
                if word.lower() in stopwords.words('english'):
                    entry['targetCaptions'][ind] = entry['targetCaptions'][ind].replace(word, '')
        filtered_data.append(entry)
    return filtered_data


def check_and_replace(tokenizer, entry, name):
    numbers = tokenizer.tokenize(entry[name])
    for num in numbers:
        entry[name] = entry[name].replace(num, '[n]')
    return entry


def replace_numbers(data):
    output = []
    tokenizer = nltk.RegexpTokenizer(r'[-+]?\d*\.\d+|\d+')
    for entry in data:
        numbers = tokenizer.tokenize(entry['postText'][0])
        for num in numbers:
            entry['postText'][0] = entry['postText'][0].replace(num, '[n]')
        entry = check_and_replace(tokenizer, entry, 'targetTitle')
        entry = check_and_replace(tokenizer, entry, 'targetDescription')
        entry = check_and_replace(tokenizer, entry, 'targetKeywords')
        for ind, par in enumerate(entry['targetParagraphs']):
            numbers = tokenizer.tokenize(par)
            for num in numbers:
                entry['targetParagraphs'][ind] = entry['targetParagraphs'][ind].replace(num, '[n]')
        for ind, par in enumerate(entry['targetCaptions']):
            numbers = tokenizer.tokenize(par)
            for num in numbers:
                entry['targetCaptions'][ind] = entry['targetCaptions'][ind].replace(num, '[n]')
        output.append(entry)
    return output


def stem_and_replace(tokenizer, stemmer, entry, name):
    words = tokenizer.tokenize(entry[name])
    for word in words:
        entry[name] = entry[name].replace(word, stemmer.stem(word.lower()))
    return entry


def apply_stemming(data):
    stemmed_data = []
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    for entry in data:
        words = tokenizer.tokenize(entry['postText'][0])
        for word in words:
            entry['postText'][0] = entry['postText'][0].replace(word, ps.stem(word.lower()))
        entry = stem_and_replace(tokenizer, entry, 'targetTitle')
        entry = stem_and_replace(tokenizer, entry, 'targetDescription')
        entry = stem_and_replace(tokenizer, entry, 'targetKeywords')
        for ind, par in enumerate(entry['targetParagraphs']):
            words = tokenizer.tokenize(par)
            for word in words:
                entry['targetParagraphs'][ind] = entry['targetParagraphs'][ind].replace(word, ps.stem(word.lower()))
        for ind, par in enumerate(entry['targetCaptions']):
            words = tokenizer.tokenize(par)
            for word in words:
                entry['targetCaptions'][ind] = entry['targetCaptions'][ind].replace(word, ps.stem(word.lower()))
        stemmed_data.append(entry)
    return stemmed_data


def link_and_replace(tokenizer, entry, name):
    links = tokenizer.tokenize(entry[name])
    for link in links:
        entry[name] = entry[name].replace(link, 'url')
    return entry


def find_links(data):
    output_data = []
    tokenizer = nltk.RegexpTokenizer(r'https?://(?:[-\w./.]|(?:%[\da-fA-F]{2}))+')
    for entry in data:
        links = tokenizer.tokenize(entry['postText'][0])
        for link in links:
            entry['postText'][0] = entry['postText'][0].replace(link, 'url')
        entry = link_and_replace(tokenizer, entry, 'targetTitle')
        entry = link_and_replace(tokenizer, entry, 'targetDescription')
        entry = link_and_replace(tokenizer, entry, 'targetKeywords')
        for ind, par in enumerate(entry['targetParagraphs']):
            links = tokenizer.tokenize(par)
            for link in links:
                entry['targetParagraphs'][ind] = entry['targetParagraphs'][ind].replace(link, 'url')
        for ind, par in enumerate(entry['targetCaptions']):
            links = tokenizer.tokenize(par)
            for link in links:
                entry['targetCaptions'][ind] = entry['targetCaptions'][ind].replace(link, 'url')
        output_data.append(entry)
    return output_data
