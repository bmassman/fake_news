#!/usr/bin/env python3
"""
Module to analyze results of article training with every combination of feature
sets.
"""
import os
import pandas as pd
import statsmodels.formula.api as smf


def linear_regression(classifier: str = None):
    """
    Output results of linear regression model of feature sets and learner
    against prediction accuracy.
    """
    results_file = os.path.join('fake_news', 'results', 'results.csv')
    df = pd.read_csv(results_file)
    if classifier:
        df = df[df['classifier'] == classifier]
    formula = ('test_accuracy ~ classifier + tags + spell + grammar + '
               '+ word_count + tfidf + lsh + title + sentiment + 0')
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())


if __name__ == '__main__':
    linear_regression('Logistic Regression')
