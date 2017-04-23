#!/usr/bin/env python3
"""
This module defines the ArticleDB class which provides a single dispatch for
creating a trainable dataset for modeling.
"""
from typing import Dict, Sequence
from copy import deepcopy
from itertools import count
from scipy.sparse import coo_matrix, csc_matrix, hstack
import pandas as pd
from .pipeline.build_df import build_df
from .pipeline.db_cleaner import clean_data
from .pipeline.db_transformer import transform_data


class ArticleDB:
    """
    Provides data structure and methods to enable training models for fake
    news detection.
    """
    def __init__(self, *, start_date: str = None,
                 end_date: str = None,
                 tfidf: bool = True,
                 author: bool = True,
                 tags: bool = True,
                 title: bool = True,
                 ngram: int = 1,
                 domain_endings: bool = True,
                 word_count: bool = True,
                 misspellings: bool = True,
                 grammar_mistakes: bool = True,
                 lshash: bool = True,
                 source_count: bool = True,
                 sentiment: bool = True,
                 stop_words: bool = False) -> None:
        """
        Initialize parameters for ArticleDB object.
        :param start_date: first date to include in article dataset with format
                           'YYYY-MMM-DD'
        :param end_date: last date to include in article dataset with format
                         'YYYY-MM-DD'
        :param tfidf: add tfidf of article text to X
        :param author: add author categorical text to X
        :param tags: add tags categorical text to X
        :param title: add title categorical text to X
        :param ngram: largest ngram to include in text and title vectorization
        :param domain_endings: add categorical for domain endings to X
        :param word_count: add word count column to X
        :param misspellings: add count of misspellings to X
        :param grammar_mistakes: add count of grammar mistakes to X
        :param lshash: add hash of tfidf to X
        :param source_count: add count of articles from the articles' source
                             to X
        :param sentiment: add sentiment scores to X
        :param stop_words: remove English stop words from text and title tfidf
                           features
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tfidf = tfidf
        self.author = author
        self.tags = tags
        self.title = title
        self.ngram = ngram
        self.domain_endings = domain_endings
        self.word_count = word_count
        self.misspellings = misspellings
        self.grammar_mistakes = grammar_mistakes
        self.lshash = lshash
        self.source_count = source_count
        self.sentiment = sentiment
        self.stop_words = stop_words
        self._X = None
        self._full_X = None
        self._y = None
        self.feature_names = None
        self._full_feature_names = None
        self.ground_truth_X = None
        self.ground_truth_y = None
        self.df = None
        self.ground_truth = None

    def _get_values(self) -> (coo_matrix, Dict[str, int], pd.Series):
        """Return sparse matrix X based on object parameters."""
        self.df = build_df('articles', self.start_date, self.end_date)
        self.ground_truth = build_df('ground_truth')
        self.df = clean_data(self.df)
        self.ground_truth = clean_data(self.ground_truth)
        res = transform_data(self.df, self.ground_truth,
                             tfidf=self.tfidf, author=self.author,
                             tags=self.tags, title=self.title,
                             ngram=self.ngram,
                             domain_endings=self.domain_endings,
                             word_count=self.word_count,
                             misspellings=self.misspellings,
                             grammar_mistakes=self.grammar_mistakes,
                             lshash=self.lshash,
                             source_count=self.source_count,
                             sentiment=self.sentiment,
                             stop_words=self.stop_words)
        (self._X, self.ground_truth_X, self.feature_names,
             self._y, self.ground_truth_y) = res
        return res

    @property
    def X(self) -> coo_matrix:
        """Getter method for X, the article database training data."""
        if self._X is None:
            self._get_values()
        return self._X

    @X.setter
    def X(self, value: coo_matrix) -> None:
        """Set the value of X."""
        self._X = value

    @X.deleter
    def X(self) -> None:
        self._X = None

    @property
    def y(self) -> pd.Series:
        """
        Return labels for articles.
        Fake news is 1, truthful news is 0.
        """
        if self._y is None:
            self._get_values()
        return self._y

    @y.setter
    def y(self, value: Sequence[int]) -> None:
        self._y = value

    @y.deleter
    def y(self):
        self._y = None

    def split_by_date(self, date: str):
        """Return two ArticleDBs by splitting current ArticleDB by date."""
        if self._X is None:
            self._get_values()
        db1 = deepcopy(self)
        db2 = deepcopy(self)
        db1_rows = (self.df['date'] < date).values
        db1.X = self._X[db1_rows]
        db1.y = self._y[db1_rows]
        db1.df = self.df.loc[db1_rows]
        db2_rows = (self.df['date'] >= date).values
        db2.X = self._X[db2_rows]
        db2.y = self._y[db2_rows]
        db2.df = self.df[db2_rows]
        return db1, db2

    def partial_X(self, **kwargs) -> None:
        """
        Set self.X to include subset of feature sets. The full value of X
        is then stored in self._full_X.
        """
        if self._X is None:
            self._get_values()
        if self._full_X is None:
            self._full_X = csc_matrix(deepcopy(self._X))
            self._full_feature_names = self.feature_names
        feature_map = {'author': 'auth',
                       'tfidf': 'text',
                       'tags': 'tag',
                       'title': 'title',
                       'domain_endings': 'domain',
                       'word_count': 'word_count',
                       'misspellings': 'misspellings',
                       'grammar_mistakes': 'grammar_mistakes',
                       'lshash': 'lsh',
                       'source_count': 'source_count',
                       'sentiment': 'sent'}
        feature_sets = set()
        for feature, include in kwargs.items():
            if not include:
                continue
            if not getattr(self, feature):
                raise ValueError('Cannot include feature that was not in'
                                 'original X.')
            feature_sets.add(feature_map[feature])
        kept_cols = []
        self.feature_names = {}
        new_col = count()
        for col, feature in self._full_feature_names.items():
            if any(feature.startswith(prefix) for prefix in feature_sets):
                kept_cols.append(col)
                self.feature_names[next(new_col)] = feature
        self._X = hstack([self._full_X.getcol(c) for c in kept_cols])

    def __repr__(self):
        db_vars = repr(self.__dict__)[1:-1]
        db_vars = db_vars.replace(': ', '=').replace("'", '')
        return f'{self.__class__}({db_vars})'


if __name__ == '__main__':
    article_db = ArticleDB(grammar_mistakes=False, misspellings=False,
                           word_count=False, lshash=False, source_count=False,
                           domain_endings=True)
    print(article_db.X.shape)
    tfidf_count = 0
    title_count = 0
    author_count = 0
    tag_count = 0
    domain_count = 0
    for feature in article_db.feature_names.values():
        if feature.startswith('text'):
            tfidf_count += 1
        elif feature.startswith('title'):
            title_count += 1
        elif feature.startswith('auth'):
            author_count += 1
        elif feature.startswith('tag'):
            tag_count += 1
        elif feature.startswith('domain'):
            domain_count += 1
    print(f'tfidf: {tfidf_count}')
    print(f'title: {title_count}')
    print(f'author: {author_count}')
    print(f'tag: {tag_count}')
    print(f'domain: {domain_count}')

