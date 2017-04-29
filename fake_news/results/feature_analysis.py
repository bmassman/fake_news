#!/usr/bin/env python3
"""
Module to analyze results of article training with every combination of feature
sets.
"""
import os
import pandas as pd
from itertools import combinations
import statsmodels.formula.api as smf
import seaborn as sns


def get_results(interactions: bool = False):
    results_file = os.path.join('fake_news', 'results', 'results.csv')
    df = pd.read_csv(results_file)
    if interactions:
        features = ['tags', 'spell', 'grammar', 'word_count', 'tfidf', 'lsh',
                    'title', 'sentiment']
        for feat_1, feat_2 in combinations(features, 2):
            df[f'{feat_1}x{feat_2}'] = df[feat_1] * df[feat_2]
    return df


def linear_regression(df, classifier: str = None):
    """
    Output results of linear regression model of feature sets and learner
    against prediction accuracy.
    """
    features = [f for f in list(df.columns.values)
                if f not in ['classifier', 'test_accuracy']]
    if classifier:
        df = df[df['classifier'] == classifier]
    else:
        features.insert(0, '0 + classifier')
    formula = f'test_accuracy ~ {"+".join(features)}'
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())


def response_surface_analysis(df):
    """Perform response surface analysis on df."""

    def tally_results(df):
        features = [f for f in list(df.columns.values)
                    if f not in ['classifier', 'test_accuracy']]
        classifiers = ['Decision Tree', 'Linear SVC', 'Logistic Regression',
                       'Multinomial NB', 'Random Forest', 'Voting Classifier']
        for classifier in classifiers:
            for feature in features:
                with_feature = df[(df[feature] == 1)
                                  & (df['classifier'] == classifier)]
                wo_feature = df[(df[feature] == 0)
                                & (df['classifier'] == classifier)]
                acc_diff = (with_feature['test_accuracy'].mean()
                            - wo_feature['test_accuracy'].mean())
                yield classifier, feature, acc_diff

    results = pd.DataFrame([res for res in tally_results(df)],
                           columns=['classifier', 'feature', 'effect'])
    class_order = list(df.groupby('classifier')['test_accuracy']
                         .mean()
                         .sort_values(ascending=False)
                         .index)
    results['classifier'] = pd.Categorical(results['classifier'],
                                           categories=class_order)
    feat_order = list(results.groupby('feature')['effect']
                             .mean()
                             .sort_values(ascending=False)
                             .index)
    results['feature'] = pd.Categorical(results['feature'],
                                        categories=feat_order)
    results.sort_values(['feature', 'classifier'], inplace=True)
    print(results)
    sns.barplot('feature', 'effect', hue='classifier', data=results)
    sns.plt.legend()
    sns.plt.show()


if __name__ == '__main__':
    df = get_results(interactions=True)
    linear_regression(df)
    df = get_results()
    response_surface_analysis(df)
