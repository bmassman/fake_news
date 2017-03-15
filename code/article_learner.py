#!/usr/bin/env python3
"""
This module trains supervised learners to predict the validity of news
articles.
"""
from typing import Type
from operator import itemgetter
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    preds = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, preds, labels=[1, 0])
    accuracy = np.mean(y_test == preds)
    learner_repr = repr(learner)[:repr(learner).find('(')]
    print(f'{learner_repr} with parameters {model.best_params_}:')
    print(f'\tval-accuracy: {model.best_score_}')
    print(f'\ttest-accuracy: {accuracy}')
    print(f'\tconfusion matrix: {conf_mat}')
    var_imp = variable_importance(model.best_estimator_)
    print_top_vars(var_imp, 50, data.feature_names)


def variable_importance(estimator: Type[ClassifierMixin]) -> np.array:
    if hasattr(estimator, 'coef_'):
        return estimator.coef_[0]
    if hasattr(estimator, 'feature_importances_'):
        return(estimator.feature_importances_)


def print_top_vars(var_imp: np.array, n: int, feature_names: dict) -> None:
    """Fetch, order, and print top n model variables."""
    top_n_vars = np.argpartition(var_imp, -n)[-n:]
    top_n_map = {}
    for feature_col in top_n_vars:
        feature_name = feature_names[feature_col]
        feature_score = var_imp[feature_col]
        top_n_map[feature_name] = feature_score
    top_n_ordered = sorted(top_n_map.items(), key=itemgetter(1), reverse=True)
    print('\tmost important features:')
    for rank, (feature_name, feature_score) in enumerate(top_n_ordered):
        print(f'\t\t{rank + 1}: {feature_name} = {feature_score}')


def article_trainers():
    """
    Run repeated models against article db to predict validity score for
    articles.
    """
    articles = ArticleDB(domain_endings=False, author=False,
                         source_count=False, start_date='2017-03-01',
                         end_date='2017-03-05')
    train_model(articles, DecisionTreeClassifier, {})
    train_model(articles, RandomForestClassifier, {})
    train_model(articles, LogisticRegression, {'C': [0.01, 0.1, 1, 10, 100]})
    train_model(articles, MultinomialNB, {'alpha': [0.1, 1.0, 10.0, 100.0]})

if __name__ == '__main__':
    article_trainers()
