#!/usr/bin/env python3
"""
Script to transform dataset to prepare for modeling.
"""
import csv
import re
from typing import Sequence, Dict, Set, List
from urllib.parse import urlparse
from collections import defaultdict
from itertools import count
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def multi_hot_encode(x: Sequence[str],
                     prefix: str) -> (coo_matrix, Dict[str, int]):
    """
    Return sparse matrix encoding categorical variables in x and dictionary
    mapping categorical variables to column numbers.
    Each record in x must be a single string with the categorical variables
    separated by a comma. The prefix prepends the categorical variable name
    to prevent collisions.
    """
    data = []
    i = []
    j = []
    col = count()
    dummy_col = defaultdict(lambda: next(col))
    for row, cat_vars in enumerate(x):
        for cat_var in cat_vars.split(','):
            prepended = f'{prefix}_{cat_var}'
            data.append(1)
            i.append(row)
            j.append(dummy_col[prepended])
    return coo_matrix((data, (i, j))), {v: k for k, v in dummy_col.items()}


def tfidf_text(x: Sequence[str],
               prefix: str,
               ngram: int = 1) -> (coo_matrix, Dict[int, str]):
    """
    Return sparse matrix encoding of TF-IDF encoding of x and dictionary
    mapping each token to a column number.
    """
    tfidf = TfidfVectorizer(ngram_range=(1, ngram))
    text = tfidf.fit_transform(x)
    token_list = tfidf.get_feature_names()
    text_map = {col: f'{prefix}_{token}'
                for col, token in enumerate(token_list)}
    return text, text_map


def combine(category_maps: Sequence[Dict[int, str]]) -> Dict[int, str]:
    """Return combined dictionary for mapping categories to column number."""
    combined = category_maps[0]
    for category_map in category_maps[1:]:
        offset = len(combined)
        offset_map = {col + offset: cat for col, cat in category_map.items()}
        combined.update(offset_map)
    return combined


def label_urls(netloc: pd.Series) -> pd.Series:
    """
    Returns Series corresponding to article labels.
    (1 is fake, 0 is true).
    """
    url_labels = defaultdict(lambda: float('nan'))
    with open('url_labels.csv', 'r') as f:
        reader = csv.reader(f)
        for domain, label in reader:
            label = float(label) if label else float('nan')
            url_labels[domain] = label
    return netloc.apply(lambda u: url_labels[u])


def get_netloc(urls: pd.Series) -> pd.Series:
    """Return series of netlocs from article urls."""
    return urls.apply(lambda u: urlparse(u).netloc)


def get_domain_ending(url: str) -> str:
    """Return ending of domain."""
    netloc = urlparse(url).netloc
    match = re.search(r'\.(.+?)$', netloc)
    return match.group(1)


def get_source_count(netlocs: pd.Series) -> coo_matrix:
    """
    Return coo_matrix corresponding to the count of articles in database from
    each article's publisher.
    """
    source_counts = netlocs.groupby(netlocs).transform('count')
    return coo_matrix(source_counts).T


def count_misspellings(text: str, dictionary: Set[str]) -> float:
    """Return proportion of misspellings in each article's text."""
    words = re.sub(r'[^A-Za-z]', ' ', text).split()
    misspellings = 0
    word_count = len(words)
    if word_count == 0:
        return 0.0
    for word in words:
        if word[0].isupper():
            continue
        if word.lower() not in dictionary:
            misspellings += 1
    return misspellings / len(words)


def get_misspellings(text: pd.Series) -> pd.Series:
    """Return Series of misspelling counts in text."""
    with open('Dictionary_690.csv', 'r') as f:
        words = f.readlines()
    words = map(lambda x: x.strip(), words)
    dictionary = {word for word in words}
    return text.apply(lambda x: count_misspellings(x, dictionary))


def get_lshash(text: coo_matrix) -> List[str]:
    """
    Return list of cosine LSHs encoding text.
    """
    def cosine_LSH(vector, planes):
        """
        Return a single cosine LSH for a particular record and given planes.
        """
        sig = 0
        for plane in planes:
            sig <<= 1
            if vector.dot(plane) >= 0:
                sig |= 1
        return str(sig)

    bits = 512
    random_projections = np.random.randn(bits, text.shape[1])
    hashes = [cosine_LSH(text.getrow(idx), random_projections)
              for idx in range(text.shape[0])]
    return hashes


def transform_data(articles: pd.DataFrame,
                   ground_truth: pd.DataFrame, *,
                   tfidf: bool,
                   author: bool,
                   tags: bool,
                   title: bool,
                   ngram: int,
                   domain_endings: bool,
                   word_count: bool,
                   misspellings: bool,
                   lshash: bool,
                   source_count: bool) -> (csr_matrix, csr_matrix,
                                           Dict[str, int],
                                           pd.Series, pd.Series):
    """
    Return sparse matrix of features for modeling and dict mapping categories
    to column numbers.
    """
    articles['netloc'] = get_netloc(articles['url'])
    ground_truth['netloc'] = get_netloc(ground_truth['url'])
    articles['labels'] = label_urls(articles['netloc'])
    ground_truth['labels'] = ground_truth['labels'].apply(pd.to_numeric)
    articles.dropna(subset=['labels'], inplace=True)
    articles_end = len(articles.index)
    articles = articles.append(ground_truth, ignore_index=True)
    res = []
    if author:
        res.append(multi_hot_encode(articles['authors'], 'auth'))
    if tags:
        res.append(multi_hot_encode(articles['tags'], 'tag'))
    if tfidf:
        tfidfed_text, tfidf_dict = tfidf_text(articles['text'], 'text', ngram)
        res.append((tfidfed_text, tfidf_dict))
    if title:
        res.append(tfidf_text(articles['title'], 'title', ngram))
    if domain_endings:
        articles['domain_ending'] = articles['url'].apply(get_domain_ending)
        res.append(multi_hot_encode(articles['domain_ending'], 'domain'))
    if word_count:
        res.append((coo_matrix(articles['word_count']).T, {0: 'word_count'}))
    if misspellings:
        articles['misspellings'] = get_misspellings(articles['text'])
        res.append((coo_matrix(articles['misspellings']).T,
                   {0: 'misspellings'}))
    if lshash:
        if not tfidf:
            tfidfed_text, _ = tfidf_text(articles['text'], 'text', ngram)
        res.append(multi_hot_encode(get_lshash(tfidfed_text), 'lsh'))
    if source_count:
        res.append((get_source_count(articles['netloc']), {0: 'source_count'}))
    features = hstack([r[0] for r in res]).tocsr()
    category_map = combine([r[1] for r in res])
    articles.drop(articles.index[articles_end:], inplace=True)
    return (features[:articles_end, :], features[articles_end:, :],
            category_map, articles['labels'], ground_truth['labels'])
