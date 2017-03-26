# -*- coding: utf-8 -*-
"""
Simple sentiment analysis

"""
import re
import nltk
import string
import pandas as pd
import unicodedata
from HTMLParser import HTMLParser

from contractions import CONTRACTION_MAP

stopword_list = nltk.corpus.stopwords.words('english')

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_html(text):
    html_stripper = MLStripper()
    html_stripper.feed(text)
    return html_stripper.get_data()

def normalize_accented_characters(text):
    text = unicodedata.normalize('NFKD',
                                 text.decode('utf-8')
                                 ).encode('ascii', 'ignore')
    return text

html_parser = HTMLParser()
def unescape_html(parser, text):
    return parser.unescape(text)

def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens 
                       if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_numbers(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens
                       if not unicode(token, 'utf-8').isnumeric()]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def stem_text(text):
    from nltk.stem import LancasterStemmer
    ls = LancasterStemmer()
    tokens = tokenize_text(text)
    filtered_tokens = [ls.stem(token) for token in tokens]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def expand_contractions(text, contraction_mapping):
    contraction_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = contraction_pattern.sub(expand_match, text)
    return expanded_text

def remove_punctuation(text):
    return text.translate(None, string.punctuation)

def removeNonAscii(text):
    return re.sub(r'[^\x00-\x7f]',r'', text)

def normalize_text(text):
    # Get rid of any html
    text = strip_html(text)
    # Remove non-ascii characters
    text = removeNonAscii(text)
    # Expand contracted words isn't -> is not
    text = expand_contractions(text, CONTRACTION_MAP)
    # Remove punctuation
    text = remove_punctuation(text)
    # Get rid of non-English letters
    text = normalize_accented_characters(text)
    # Get rid of special characters hidden in the text
    text = remove_special_characters(text)
    # Lemmatize text better -> good not bet
    # To be done
    # all text to lower case
    text = text.lower()
    # Remove all numbers that are just numbers
    text = remove_numbers(text)
    # Remove stop words
    text = remove_stopwords(text)
    # Stem text - not sure if we need to do this
    #text = stem_text(text)
    return(text)

def sentiments():
    filename = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    # read in sentiments
    dataset = pd.read_csv(filename, 
                          sep = '\t', 
                          header = None, 
                          names = ['Word', 'Affect', 'Association'])
    # Only care if the sentiments have an association
    return dataset[dataset['Association'] == 1][['Word', 'Affect']]

    
def count_affect(text):
    '''
    text should be cleaned before running this
    '''
    
    sentiment = sentiments()
    Affects = sentiment.Affect.unique()
    tokens = tokenize_text(text)
    counts = {}
    wc = len(tokens)
    counts['wc'] = wc
    for affect in Affects:
        Affect_List = set(sentiment.Word[sentiment.Affect == affect])
        num = sum([(token in Affect_List) for token in tokens])
        counts[affect] = num
    counts['neutral'] = counts['wc'] - counts['positive'] - counts['negative']
    return counts
    
if __name__ == '__main__':
    # example usage
    # Read in some text - change file name as appropriate
    with open("myText.txt", 'r') as myfile:
        text = myfile.read().replace('\n', '')
    text = normalize_text(text)
    score = count_affect(text)
    print score
