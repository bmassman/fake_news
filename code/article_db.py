#!/usr/bin/env python3
"""
This module defines the ArticleDB class which provides a single dispatch for
creating a trainable dataset for modeling.
"""
from typing import Dict, Sequence
from scipy.sparse import coo_matrix
from code.db_transformer import transform_data


class ArticleDB:
    """
    Provides data structure and methods to enable training models for fake
    news detection.
    """
    def __init__(self, *,
                 tfidf: bool = True,
                 author: bool = True,
                 tags: bool = True,
                 title: bool = True,
                 ngram: int = 1,
                 domain_endings: bool = True,
                 word_count: bool = True,
                 misspellings: bool = True,
                 source_count: bool = True) -> None:
        """
        Initialize parameters for ArticleDB object.
        :param tfidf: add tfidf of article text to X
        :param author: add author categorical text to X
        :param tags: add tags categorical text to X
        :param title: add title categorical text to X
        :param ngram: largest ngram to include in text and title vectorization
        :param domain_endings: add categorical for domain endings to X
        :param word_count: add word count column to X
        :param misspellings: add count of misspellings to X
        :param source_count: add count of articles from the articles' source
                             to X
        """
        self.tfidf = tfidf
        self.author = author
        self.tags = tags
        self.title = title
        self.ngram = ngram
        self.domain_endings = domain_endings
        self.word_count = word_count
        self.misspellings = misspellings
        self.source_count = source_count
        self._X = None
        self._y = None
        self.column_number = None

    def _get_values(self) -> (coo_matrix, Dict[str, int], coo_matrix):
        """Return sparse matrix X based on object parameters."""
        res = transform_data(tfidf=self.tfidf, author=self.author,
                             tags=self.tags, title=self.title,
                             ngram=self.ngram,
                             domain_endings=self.domain_endings,
                             word_count=self.word_count,
                             misspellings=self.misspellings,
                             source_count=self.source_count)
        return res

    @property
    def X(self) -> coo_matrix:
        """Getter method for X, the article database training data."""
        if self._X is None:
            self._X, self.column_number, self._y = self._get_values()
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
    def y(self) -> coo_matrix:
        """
        Return labels for articles.
        Fake news is 1, truthful news is 0.
        """
        if self._y is None:
            self._X, self.column_number, self._y = self._get_values()
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
    article_db = ArticleDB()
    print(repr(article_db))
    X = article_db.X
    print(X)
    y = article_db.y
    print(y)
