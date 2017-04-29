#!/usr/bin/env python3
"""
This module trains supervised learners to predict the validity of news
articles.
"""
from typing import Type, Sequence, Optional
from operator import itemgetter
from random import sample
import math
from itertools import product
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
from fake_news.article_db import ArticleDB


def train_model(data: ArticleDB,
                learner: Type[ClassifierMixin],
                param_grid: dict, *,
                test_articles: Optional[ArticleDB] = None,
                most_important_features: bool = False,
                examples: bool = False,
                ground_truth_as_test: bool = False,
                probabilities: bool = False) -> ClassifierMixin:
    """Trains classifier learner on data and reports test set accuracy."""
    if ground_truth_as_test and test_articles:
        raise ValueError('ground_truth_as_test must be False if test_articles'
                         'are supplied')
    if callable(learner):
        learner = learner()
    X, y = data.X, data.y
    if ground_truth_as_test or test_articles:
        X_train = X
        y_train = y
    if ground_truth_as_test:
        X_test = data.ground_truth_X
        y_test = data.ground_truth_y
        df_test = data.ground_truth
    elif test_articles:
        X_test = test_articles.X
        y_test = test_articles.y
        df_test = test_articles.df
    else:
        X_train, X_test, y_train, y_test, df_train, df_test = (
            train_test_split(X, y, data.df, test_size=0.2))
    model = GridSearchCV(learner, param_grid).fit(X_train, y_train)
    best_model = model.best_estimator_
    preds = best_model.predict(X_test)
    conf_mat = confusion_matrix(y_test, preds, labels=[1, 0])
    accuracy = np.mean(y_test == preds)
    learner_repr = repr(learner)[:repr(learner).find('(')]
    print(f'{learner_repr} with parameters {model.best_params_}:')
    print(f'\tval-accuracy: {model.best_score_}')
    print(f'\ttest-accuracy: {accuracy}')
    print(f'\tconfusion matrix: [{conf_mat[0]}')
    print(f'\t                   {conf_mat[1]}]')
    var_imp = variable_importance(model.best_estimator_)
    if most_important_features:
        print_top_vars(var_imp, 50, data.feature_names)
    if examples:
        article_examples(df_test, y_test, preds)
    if probabilities and hasattr(best_model, 'predict_proba'):
        test_probabilities(best_model, X_test, y_test)
    return best_model


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


def test_probabilities(model: ClassifierMixin, X: np.array, y: pd.Series,
                       bins: int = 10, threshold: float = 0.5):
    """Print confusion matrix based on class probability."""
    probs = [p[1] for p in model.predict_proba(X)]
    print('\tProbabilities')
    df = pd.DataFrame({'prob': probs, 'label': y})
    step = 1 / bins
    cut_labels = [round(step * f, 1) for f in range(10)]
    by_prob = (df.groupby(pd.cut(df['prob'], bins, labels=cut_labels))
                 .agg(['sum', 'count'])['label'])
    print('\t\tprobs\t1\t0\tacc')
    for index, row in by_prob.iloc[::-1].iterrows():
        ones = row['sum']
        if math.isnan(ones):
            ones = 0
        else:
            ones = int(ones)
        count = row['count']
        zeros = int(count) - ones
        if count > 0:
            acc = zeros / count if index < threshold else ones / count
        else:
            acc = 0.0
        print(f'\t\t{index}\t{ones}\t{zeros}\t{acc:.3f}')


def article_trainers(articles: ArticleDB):
    """
    Run repeated models against article db to predict validity score for
    articles.
    """
    models = [(DecisionTreeClassifier, {}),
              (RandomForestClassifier, {}),
              (LogisticRegression, {'C': [0.01, 0.1, 1, 10, 100]}),
              (MultinomialNB, {'alpha': [0.1, 1.0, 10.0, 100.0]}),
              (LinearSVC, {'C': [0.01, 0.1, 1, 10, 100]})]
    trained_models = []
    for classifier, param_grid in models:
        res = train_model(articles, classifier, param_grid, probabilities=True)
        trained_models.append((str(res), res))
    ensemble_learner = VotingClassifier(estimators=trained_models[:4],
                                        voting='soft')
    train_model(articles, ensemble_learner, {})


def feature_analysis():
    articles = ArticleDB(domain_endings=False,
                         author=False,
                         source_count=False,
                         tags=True,
                         misspellings=True,
                         grammar_mistakes=False,
                         word_count=False,
                         tfidf=True,
                         ngram=1,
                         lshash=False,
                         title=True,
                         sentiment=False,
                         stop_words=False)
    settings = product((True, False), repeat=8)
    key_words = ['tags', 'misspellings', 'grammar_mistakes', 'word_count',
                 'tfidf', 'lshash', 'title', 'sentiment']
    for setting in settings:
        kwargs = dict(zip(key_words, setting))
        print(kwargs)
        articles.partial_X(**kwargs)
        article_trainers(articles)


def time_slices():
    """Show results for time slices of article trainers."""
    model = LogisticRegression
    test_dates = [('2017-03-01', '2017-03-02'),
                  ('2017-03-01', '2017-03-08'),
                  ('2017-03-01', '2017-03-15'),
                  ('2017-03-01', '2017-03-22'),
                  ('2017-03-01', '2017-03-29')]
    for start, end in test_dates:
        print(f'From {start} to {end}')
        articles = ArticleDB(start_date=start, end_date=end,
                             domain_endings=False,
                             author=False,
                             source_count=False,
                             tags=True,
                             misspellings=True,
                             grammar_mistakes=False,
                             word_count=False,
                             tfidf=True,
                             ngram=1,
                             lshash=False,
                             title=True,
                             sentiment=False,
                             stop_words=False)
        train, test = articles.split_by_date(end)
        train_model(train, model, {'C': [100]}, test_articles=test)


if __name__ == '__main__':
    time_slices()
