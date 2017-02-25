#!/usr/bin/env python3
"""
Script to produce summary statistics for news articles.
"""
import sqlite3
import re
from contextlib import closing
from urllib.parse import urlparse
import pandas as pd
import seaborn as sns

DB_FILE_NAME = 'articles.db'


def get_url_base(row):
    """Return base url from article row."""
    url = row['url']
    domain_ending = re.compile(r'\.[a-z]{2,3}$')
    domain_prefix = re.compile(r'^[a-z]+\.')
    domains_to_keep = {'wsj', 'cnn', 'cbs', 'nbc', 'bbc', 'de'}

    net_loc = urlparse(url).netloc
    while net_loc.startswith('www.'):
        net_loc = net_loc[4:]
    while True:
        match = domain_ending.search(net_loc)
        if not match or match.group(0) in domains_to_keep:
            break
        net_loc = domain_ending.sub('', net_loc)
    net_loc = domain_prefix.sub('', net_loc)
    return net_loc


def count_words(row):
    """Count words in text from article row."""
    text = row['text']
    return len(text.split())


def build_df() -> pd.DataFrame:
    """Build dataframe with derived fields."""
    with closing(sqlite3.connect(DB_FILE_NAME)) as conn:
        articles = pd.read_sql_query('select * from articles', conn)

    articles['base_url'] = articles.apply(get_url_base, axis=1)
    articles['word_count'] = articles.apply(count_words, axis=1)
    return articles


def print_full(x):
    """Print all rows in Pandas DataFrame x."""
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def global_stats(articles: pd.DataFrame):
    """Calculate global stats on article db."""
    print(f'Number of articles: {len(articles):,}')
    num_sources = len(pd.value_counts(articles['base_url'], sort=False))
    print(f'Number of news sources: {num_sources}')
    mean_wc = articles['word_count'].mean()
    print(f'Global mean word count: {mean_wc:.1f}')
    missing_authors = (articles['authors'] == '').sum()
    print(f'Missing authors: {missing_authors:,}')
    missing_titles = (articles['title'] == '').sum()
    print(f'Missing titles: {missing_titles}')
    missing_texts = (articles['text'] == '').sum()
    print(f'Missing texts: {missing_texts:,}')


def calculate_word_count_stats(articles: pd.DataFrame):
    """Calculate aggregate word count statistics on each source's articles."""
    by_source = articles.groupby(['base_url'])['word_count']
    by_source = by_source.agg(['count', 'mean', 'std'])
    by_source.sort_values('count', ascending=False, inplace=True)
    print_full(by_source)

    top_sources = by_source.head(10).index
    sns.boxplot(x='base_url', y='word_count',
                data=articles[articles['base_url'].isin(top_sources)])
    sns.plt.show()


def calculate_missing_values(articles: pd.DataFrame):
    """Calculate count of nulls in each column."""
    def null_fields(x: pd.Series) -> pd.Series:
        return pd.Series({'no_author': (x['authors'] == '').sum(),
                          'no_text': (x['text'] == '').sum(),
                          'no_title': (x['title'] == '').sum()})

    null_field_count = articles.groupby('base_url').apply(null_fields)
    null_field_count = null_field_count[(null_field_count.T != 0).any()]
    print_full(null_field_count)


def show_stats():
    """Display statistics on articles."""
    articles = build_df()
    global_stats(articles)
    calculate_word_count_stats(articles)
    calculate_missing_values(articles)
    sns.kdeplot(articles['word_count'], bw=1)
    sns.plt.show()


if __name__ == '__main__':
    show_stats()
