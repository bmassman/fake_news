#!/usr/bin/env python3
"""
This module trains supervised learners to predict the validity of news
articles.
"""
from typing import Type
from operator import itemgetter
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from code.article_db import ArticleDB


def train_model(data: ArticleDB,
                learner: Type[ClassifierMixin],
                param_grid: dict) -> None:
    """Trains classifier learner on data and reports test set accuracy."""
    learner = learner()
    X, y = data.X, data.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GridSearchCV(learner, param_grid).fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    learner_repr = repr(learner)[:repr(learner).find('(')]
    print(f'{learner_repr} with parameters {model.best_params_}:')
    print(f'\tval-accuracy: {model.best_score_}')
    print(f'\ttest-accuracy: {accuracy}')
    variable_importance(model.best_estimator_, data.feature_names)


def variable_importance(estimator: Type[ClassifierMixin],
                        feature_names: dict) -> np.array:
    if hasattr(estimator, 'coef_'):
        coef = estimator.coef_
        for var_imp in coef:
            print_top_vars(var_imp, 10, feature_names)
    if hasattr(estimator, 'feature_importances_'):
        print_top_vars(estimator.feature_importances_, 10, feature_names)


def print_top_vars(var_imp: np.array, n: int, feature_names: dict) -> None:
    """Fetch, order, and print top n model variables."""
    top_10_vars = np.argpartition(var_imp, -n)[-n:]
    top_10_map= {}
    for feature_col in top_10_vars:
        feature_name = feature_names[feature_col]
        feature_score = var_imp[feature_col]
        top_10_map[feature_name] = feature_score
    top_10_ordered = sorted(top_10_map.items(), key=itemgetter(1),
                            reverse=True)
    print('\tmost important features:')
    for rank, (feature_name, feature_score) in enumerate(top_10_ordered):
        print(f'\t\t{rank + 1}: {feature_name} {feature_score}')


def article_trainers():
    """
    Run repeated models against article db to predict validity score for
    articles.
    """
    print('Getting Data...')
    articles = ArticleDB(domain_endings=False, author=False,
                         source_count=False, start_date='2017-03-01',
                         end_date='2017-03-05')
    articles.X
    print('Starting Training')
    train_model(articles, DecisionTreeClassifier, {})
    train_model(articles, LogisticRegression, {'C': [0.01, 0.1, 1, 10, 100]})
    train_model(articles, MultinomialNB, {'alpha': [0.1, 1.0, 10.0, 100.0]})

if __name__ == '__main__':
    article_trainers()
