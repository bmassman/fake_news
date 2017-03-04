#!/usr/bin/env python3
"""
Script to transform dataset to prepare for modeling.
"""
from typing import Sequence, Dict
from urllib.parse import urlparse
from collections import defaultdict
from itertools import count
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from code.db_cleaner import clean_data


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
    return coo_matrix((data, (i, j))), dict(dummy_col)


def tfidf_text(x: Sequence[str],
               prefix: str,
               ngram: int = 1) -> (coo_matrix, Dict[str, int]):
    """
    Return sparse matrix encoding of TF-IDF encoding of x and dictionary
    mapping each token to a column number.
    """
    tfidf = TfidfVectorizer(ngram_range=(1, ngram))
    text = tfidf.fit_transform(x)
    token_list = tfidf.get_feature_names()
    text_map = {f'{prefix}_{token}': col
                for col, token in enumerate(token_list)}
    return text, text_map


def combine(category_maps: Sequence[Dict[str, int]]) -> Dict[str, int]:
    """Return combined dictionary for mapping categories to column number."""
    combined = category_maps[0]
    for category_map in category_maps[1:]:
        offset = len(combined)
        offset_map = {cat: col + offset for cat, col in category_map.items()}
        combined.update(offset_map)
    return combined


def ends_with_dotcom(url: str) -> int:
    netloc = urlparse(url).netloc
    return netloc.endswith('.com')


def transform_data(*, tfidf: bool,
                   author: bool,
                   tags: bool,
                   title: bool,
                   ngram: int,
                   is_dotcom: bool,
                   word_count: bool,
                   misspellings: bool) -> (coo_matrix, Dict[str, int]):
    """
    Return sparse matrix of features for modeling and dict mapping categories
    to column numbers.
    """
    articles = clean_data()
    res = []
    if author:
        res.append(multi_hot_encode(articles['authors'], 'auth'))
    if tags:
        res.append(multi_hot_encode(articles['tags'], 'tag'))
    if tfidf:
        res.append(tfidf_text(articles['text'], 'text', ngram))
    if title:
        res.append(tfidf_text(articles['title'], 'title', ngram))
    if is_dotcom:
        articles['dotcom'] = articles['url'].apply(ends_with_dotcom)
        res.append((coo_matrix(articles['dotcom']).T, {'is_dotcom': 0}))
    if word_count:
        res.append((coo_matrix(articles['word_count']).T, {'word_count': 0}))
    if misspellings:
        ...
    features = hstack([r[0] for r in res])
    category_map = combine([r[1] for r in res])
    return features, category_map


if __name__ == '__main__':
    X, col_map = transform_data(tfidf=False, author=True, tags=False,
                                title=True, ngram=1)
    print(X.shape)
    print(len(col_map))
