#!/usr/bin/env python3
"""
This module trains supervised learners to predict the validity of news
articles.
"""
from typing import Type, Sequence
from operator import itemgetter
from random import sample
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from fake_news.article_db import ArticleDB


def train_model(data: ArticleDB,
                learner: Type[ClassifierMixin],
                param_grid: dict,
                examples: bool = False,
                ground_truth_as_test: bool = False) -> None:
    """Trains classifier learner on data and reports test set accuracy."""
    learner = learner()
    X, y = data.X, data.y
    if ground_truth_as_test:
        X_train, X_test, y_train, y_test, df_train, df_test = (
            X, data.ground_truth_X, y, data.ground_truth_y, data.df,
            data.ground_truth)
    else:
        X_train, X_test, y_train, y_test, df_train, df_test = (
            train_test_split(X, y, data.df, test_size=0.2))
    model = GridSearchCV(learner, param_grid).fit(X_train, y_train)
    preds = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, preds, labels=[1, 0])
    accuracy = np.mean(y_test == preds)
    learner_repr = repr(learner)[:repr(learner).find('(')]
    print(f'{learner_repr} with parameters {model.best_params_}:')
    print(f'\tval-accuracy: {model.best_score_}')
    print(f'\ttest-accuracy: {accuracy}')
    print(f'\tconfusion matrix: [{conf_mat[0]}')
    print(f'\t                   {conf_mat[1]}]')
    var_imp = variable_importance(model.best_estimator_)
    print_top_vars(var_imp, 50, data.feature_names)
    if examples:
        article_examples(df_test, y_test, preds)


def variable_importance(estimator: Type[ClassifierMixin]) -> np.array:
    """Return variable importances for estimator."""
    if hasattr(estimator, 'coef_'):
        return estimator.coef_[0]
    if hasattr(estimator, 'feature_importances_'):
        return estimator.feature_importances_


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


def article_examples(test_articles: pd.DataFrame,
                     true_label: Sequence[int],
                     pred_label: Sequence[int]) -> None:
    """
    Print examples of TP, FP, TN, and FN classifications from trained model.
    """
    tp_idx = np.logical_and(true_label == 1, pred_label == 1)
    true_positives = test_articles[tp_idx]
    fp_idx = np.logical_and(true_label == 0, pred_label == 1)
    false_positives = test_articles[fp_idx]
    tn_idx = np.logical_and(true_label == 0, pred_label == 0)
    true_negatives = test_articles[tn_idx]
    fn_idx = np.logical_and(true_label == 1, pred_label == 0)
    false_negatives = test_articles[fn_idx]
    article_example_printer('True Positives', true_positives)
    article_example_printer('False Positives', false_positives)
    article_example_printer('True Negatives', true_negatives)
    article_example_printer('False Negatives', false_negatives)


def article_example_printer(category: str,
                            df: pd.DataFrame,
                            k: int = 5) -> None:
    """Print a random selection of n articles from df."""
    print(f'\t{category}')
    num_rows = len(df.index)
    if num_rows > k:
        df = df.iloc[sample(range(num_rows), k)]
    for row in df.itertuples():
        text = row.text.replace('\n', ' ')[:150]
        print(f'\t\tTitle: {row.title}')
        print(f'\t\tUrl: {row.url}')
        print(f'\t\tText: {text}\n')


def article_trainers():
    """
    Run repeated models against article db to predict validity score for
    articles.
    """
    articles = ArticleDB(domain_endings=False,
                         author=False,
                         source_count=False,
                         tags=False,
                         misspellings=True,
                         grammar_mistakes=True,
                         word_count=True,
                         tfidf=True,
                         ngram=1,
                         lshash=False,
                         title=True,
                         start_date='2017-03-01',
                         end_date='2017-03-15')
    models = [(DecisionTreeClassifier, {}),
              (RandomForestClassifier, {}),
              (LogisticRegression, {'C': [0.01, 0.1, 1, 10, 100]}),
              (MultinomialNB, {'alpha': [0.1, 1.0, 10.0, 100.0]}),
              (LinearSVC, {'C': [0.01, 0.1, 1, 10, 100]})]
    for classifier, param_grid in models:
        train_model(articles, classifier, param_grid, examples=True,
                    ground_truth_as_test=True)

if __name__ == '__main__':
    article_trainers()
