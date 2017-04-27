#!/usr/bin/env python3
"""
Module to analyze results of article training with every combination of feature
sets.
"""
import os
import pandas as pd
import statsmodels.formula.api as smf


def get_results():
    results_file = os.path.join('fake_news', 'results', 'results.csv')
    return pd.read_csv(results_file)


def linear_regression(df, classifier: str = None):
    """
    Output results of linear regression model of feature sets and learner
    against prediction accuracy.
    """
    if classifier:
        df = df[df['classifier'] == classifier]
    formula = ('test_accuracy ~ tags + spell + grammar + '
               '+ word_count + tfidf + lsh + title + sentiment + 0')
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())


def response_surface_analysis(df):
    features = ['tags', 'spell', 'grammar', 'word_count', 'tfidf', 'lsh',
                'title', 'sentiment']
    for feature in features:
        with_feature = df[df[feature] == 1]
        wo_feature = df[df[feature] == 0]
        acc_diff = (with_feature['test_accuracy'].mean()
                    - wo_feature['test_accuracy'].mean())
        print(f'{feature}: {acc_diff}')


if __name__ == '__main__':
    df = get_results()
    linear_regression(df, 'Logistic Regression')
    response_surface_analysis(df)
