#!/usr/bin/env python3
"""
Script to transform dataset to prepare for modeling.
"""
import csv
import re
from typing import Sequence, Dict
from urllib.parse import urlparse
from collections import defaultdict
from itertools import count
import pandas as pd
import random
from scipy.sparse import coo_matrix, hstack
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


def transform_data(articles, *, tfidf: bool,
                   author: bool,
                   tags: bool,
                   title: bool,
                   ngram: int,
                   domain_endings: bool,
                   word_count: bool,
                   misspellings: bool,
                   lshash: bool,
                   source_count: bool) -> (coo_matrix, Dict[str, int],
                                           coo_matrix):
    """
    Return sparse matrix of features for modeling and dict mapping categories
    to column numbers.
    """
    articles['netloc'] = get_netloc(articles['url'])
    articles['labels'] = label_urls(articles['netloc'])
    articles.dropna(subset=['labels'], inplace=True)
    res = []
    if author:
        res.append(multi_hot_encode(articles['authors'], 'auth'))
    if tags:
        res.append(multi_hot_encode(articles['tags'], 'tag'))
    if tfidf:
        res.append(tfidf_text(articles['text'], 'text', ngram))
    if title:
        res.append(tfidf_text(articles['title'], 'title', ngram))
    if domain_endings:
        articles['domain_ending'] = articles['url'].apply(get_domain_ending)
        res.append(multi_hot_encode(articles['domain_ending'], 'domain'))
    if word_count:
        res.append((coo_matrix(articles['word_count']).T, {0: 'word_count'}))
    if misspellings:
        ...
    if lshash:
        ...
        # res.append((get_lshash(articles['text']), {0: 'lshash'}))
    if source_count:
        res.append((get_source_count(articles['netloc']), {0: 'source_count'}))
    features = hstack([r[0] for r in res])
    category_map = combine([r[1] for r in res])
    return features, category_map, articles['labels']


def get_lshash (text: pd.Series) -> coo_matrix:
    """
    Return sparse matrix encoding of LSH encoding of x and dictionary
    mapping each token to a column number.
    """
    # initialise a new hash table for each hash function
    k = 1024
    d = 5
    l = 64
    lsh = LSHIndex(CosineHashFamily(d), k, l)
    lsh.size(l)
    LS_hash = lsh.index(text)
    print (LS_hash)
    return coo_matrix(LS_hash)

def dot(u,v):
    return sum(ux*vx for ux,vx in zip(u,v))

class CosineHashFamily:

    def __init__(self,d):
        self.d = d

    def create_hash_func(self):
        # each CosineHash is initialised with a random projection vector
        return CosineHash(self.rand_vec())

    def rand_vec(self):
        return [random.gauss(0,1) for i in range(self.d)]

    def combine(self,hashes):
        """ combine by treating as a bitvector """
        return sum(2**i if h > 0 else 0 for i,h in enumerate(hashes))

class CosineHash:

    def __init__(self,r):
        self.r = r

    def hash(self,vec):
        return self.sgn(dot(vec,self.r))

    def sgn(self,x):
        return int(x>0)

class LSHIndex:

    def __init__(self,hash_family,k,l):
        self.hash_family = hash_family
        self.k = k
        self.l = l
        self.hash_tables = []
        self.size(l)

    def size(self,l):
        """ update the number of hash tables to be used """
        # initialise a the hash table for the requested function
        hash_funcs = [[self.hash_family.create_hash_func() for h in range(self.k)] for l in range(self.l,l)]
        self.hash_tables.extend([(g,defaultdict(lambda:[])) for g in hash_funcs])

    def index(self,points):
        """ index the supplied points """
        self.points = points
        for g,table in self.hash_tables:
            for ix,p in enumerate(self.points):
                table[self.hash(g,p)].append(ix)

    def hash(self,g,p):
        return self.hash_family.combine([h.hash(p) for h in g])
