#!/usr/bin/env python3
"""
Script to transform dataset to prepare for modeling.
"""
from typing import Sequence, Dict
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


def vectorize_text(x: Sequence[str],
                   prefix: str) -> (coo_matrix, Dict[str, int]):
    """
    Return sparse matrix encoding of TF-IDF encoding of x and dictionary
    mapping each token to a column number.
    """
    tfidf = TfidfVectorizer()
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


def transform_data() -> (coo_matrix, Dict[str, int]):
    """
    Return sparse matrix of features for modeling and dict mapping categories
    to column numbers.
    """
    articles = clean_data()
    authors, author_map = multi_hot_encode(articles['authors'], 'auth')
    tags, tag_map = multi_hot_encode(articles['tags'], 'tag')
    text, text_map = vectorize_text(articles['text'], 'text')
    features = hstack([authors, tags, text])
    category_map = combine([author_map, tag_map, text_map])
    return features, category_map


if __name__ == '__main__':
    X, col_map = transform_data()
    print(X)
    print(col_map)
