#!/usr/bin/env python3
"""
Script to produce summary statistics for news articles.
"""
import pandas as pd
import seaborn as sns
from .article_db import ArticleDB


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
    top_counts = by_source.reset_index()[by_source.index.isin(top_sources)]
    sns.barplot(x='base_url', y='count', data=top_counts)
    sns.plt.show()
    sns.boxplot(x='base_url', y='word_count',
                data=articles[articles['base_url'].isin(top_sources)])
    sns.plt.show()


def show_articles_by_source(articles: pd.DataFrame):
    """Show boxplot comparing articles by source for fake and true news."""
    by_source = (articles.groupby(['base_url', 'labels'])
                         .size()
                         .reset_index(name='count'))
    by_source = by_source[by_source['count'] > 100]
    sns.boxplot(x='labels', y='count', data=by_source)
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


def word_count_by_label(articles: pd.DataFrame):
    """Show graph of word counts by article label."""
    palette = sns.color_palette(palette='hls', n_colors=2)
    true_news_wc = articles[articles['labels'] == 0]['word_count']
    fake_news_wc = articles[articles['labels'] == 1]['word_count']
    sns.kdeplot(true_news_wc, bw=3, color=palette[0], label='True News')
    sns.kdeplot(fake_news_wc, bw=3, color=palette[1], label='Fake News')
    sns.plt.legend()
    sns.plt.show()


def show_stats():
    """Display statistics on articles."""
    articles = ArticleDB(tfidf=False, author=False, tags=False, title=False,
                         domain_endings=False, grammar_mistakes=False,
                         word_count=True, misspellings=False, lshash=False,
                         source_count=False, sentiment=False)
    articles.X
    articles = articles.df
    global_stats(articles)
    calculate_word_count_stats(articles)
    calculate_missing_values(articles)
    word_count_by_label(articles)
    show_articles_by_source(articles)


if __name__ == '__main__':
    show_stats()
