# -*- coding: utf-8 -*-
"""
Simple sentiment analysis

"""
import re
import nltk
import string
import pandas as pd
import unicodedata
from typing import Dict, Set, List
from .contractions import CONTRACTION_MAP

stopword_list = nltk.corpus.stopwords.words('english')


def removeNonAscii(text):
    return re.sub(r'[^\x00-\x7f]', '', text)


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def normalize_accented_characters(text):
    text = unicodedata.normalize('NFKD', text)
    return text.encode('ascii', 'ignore').decode('utf-8')


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_numbers(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if not token.isnumeric()]
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


def expand_contractions(text, contraction_map):
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        try:
            expanded_contraction = contraction_map[match]
        except KeyError:
            expanded_contraction = contraction_map[match.lower()]
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    contraction_pattern = re.compile(f'({"|".join(contraction_map.keys())})',
                                     flags=re.IGNORECASE|re.DOTALL)
    expanded_text = contraction_pattern.sub(expand_match, text)
    return expanded_text


def remove_punctuation(text: str,
                       translator=str.maketrans('', '', string.punctuation)):
    """Remove all punctuation from text."""
    return text.translate(translator)


def normalize_text(text):
    text = removeNonAscii(text)
    text = expand_contractions(text, CONTRACTION_MAP)
    text = remove_punctuation(text)
    text = normalize_accented_characters(text)
    text = text.lower()
    text = remove_numbers(text)
    text = remove_stopwords(text)
    return text


def sentiments():
    filename = ('fake_news/pipeline/sentiment/'
                'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    dataset = pd.read_csv(filename,
                          sep = '\t', 
                          header = None, 
                          names = ['Word', 'Affect', 'Association'])
    # Only care if the sentiments have an association
    return dataset[dataset['Association'] == 1][['Word', 'Affect']]

    
def count_affect(text: str,
                 affect_sets: Dict[str, Set[str]]) -> List[float]:
    """
    text should be cleaned before running this
    """
    tokens = tokenize_text(text)
    counts = []
    wc = len(tokens)
    if not wc:
        return [0.0] * 10
    for affect in affect_sets:
        count = sum((token in affect_sets[affect]) for token in tokens)
        counts.append(count / wc)
    return counts


def get_sentiment(text: str, affects: Dict[str, Set[str]]) -> Dict[str, float]:
    """Return dictionary of proportion of words related to each affect."""
    text = normalize_text(text)
    score = count_affect(text, affects)
    return score


def get_affect_set() -> Dict[str, Set[str]]:
    """Return dictionary of words related to each affect."""
    sentiment = sentiments()
    affects = {affect: set(sentiment.Word[sentiment.Affect == affect])
               for affect in sentiment.Affect.unique()}
    return affects


if __name__ == '__main__':
    samp = "bad. bad. happy. angry. crying. can't take it any more. Good Bye!"
    samp = normalize_text(samp)
    print(samp)
    affect_set = get_affect_set()
    score = get_sentiment(samp, affect_set)
    print(score)
