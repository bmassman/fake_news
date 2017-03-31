#!/usr/bin/env python3
"""
This module defines the ArticleDB class which provides a single dispatch for
creating a trainable dataset for modeling.
"""
from typing import Dict, Sequence
from scipy.sparse import coo_matrix
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
                 lshash: bool = True,
                 source_count: bool = True) -> None:
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
        :param lshash: add hash of tfidf to X
        :param source_count: add count of articles from the articles' source
                             to X
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
        self.lshash = lshash
        self.source_count = source_count
        self._X = None
        self._y = None
        self.feature_names = None
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
                             lshash=self.lshash,
                             source_count=self.source_count)
        return res

    @property
    def X(self) -> coo_matrix:
        """Getter method for X, the article database training data."""
        if self._X is None:
            (self._X, self.ground_truth_X, self.feature_names,
             self._y, self.ground_truth_y) = self._get_values()
        return self._X

    @X.setter
    def X(self, value: coo_matrix) -> None:
        """Set the value of X."""
        if isinstance(value, coo_matrix):
            self._X = value
        else:
            raise ValueError('X must be set to a coo_matrix.')

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
            (self._X, self.ground_truth_X, self.feature_names,
             self._y, self.ground_truth_y) = self._get_values()
        return self._y

    @y.setter
    def y(self, value: Sequence[int]) -> None:
        self._y = value

    @y.deleter
    def y(self):
        self._y = None

    def __repr__(self):
        db_vars = repr(self.__dict__)[1:-1]
        db_vars = db_vars.replace(': ', '=').replace("'", '')
        return f'{self.__class__}({db_vars})'


if __name__ == '__main__':
    article_db = ArticleDB(start_date='01-03-2017', end_date='02-03-2017')
    X = article_db.X
    y = article_db.y
    print(article_db.feature_names)
