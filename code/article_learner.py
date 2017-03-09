#!/usr/bin/env python3
"""
This module trains supervised learners to predict the validity of news
articles.
"""
from typing import Type
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
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
    variable_importance(model.best_estimator_)


def variable_importance(estimator: Type[ClassifierMixin]) -> None:
    if hasattr(estimator, 'coef_'):
        print(f'\tmodel coefficients: {estimator.coef_}')
    if hasattr(estimator, 'feature_importances_'):
        print(f'\tfeature importance: {estimator.feature_importances_}')


def article_trainers():
    """
    Run repeated models against article db to predict validity score for
    articles.
    """
    articles = ArticleDB(domain_endings=False, author=False)
    train_model(articles, DecisionTreeClassifier, {})
    train_model(articles, MultinomialNB, {'alpha': [0.1, 1.0, 10.0, 100.0]})
    train_model(articles, LogisticRegression, {'C': [0.01, 0.1, 1, 10, 100]})

if __name__ == '__main__':
    article_trainers()
