#!/usr/bin/env python3
"""
Script to produce summary statistics for news articles.
"""
import sqlite3
import re
import numpy as np
import seaborn as sns
from contextlib import closing
from collections import defaultdict
from urllib.parse import urlparse
from operator import itemgetter
from statistics import mean, stdev
from itertools import chain

DB_FILE_NAME = 'articles.db'
DOMAIN_ENDING = re.compile(r'\.[a-z]{2,3}$')
DOMAIN_PREFIX = re.compile(r'^[a-z]+\.')


def get_articles(curs):
    """Return cursor with all articles selected."""
    curs.execute('SELECT * FROM articles')
    return curs.fetchall()


def get_url_base(url):
    """Updates count of articles by news source."""
    net_loc = urlparse(url).netloc
    if net_loc.startswith('www.'):
        net_loc = net_loc[4:]
    while True:
        match = DOMAIN_ENDING.search(net_loc)
        if not match or match.group(0) == '.wsj':
            break
        net_loc = DOMAIN_ENDING.sub('', net_loc)
    net_loc = DOMAIN_PREFIX.sub('', net_loc)
    return net_loc


def calculate_stats():
    """Print statistics on article corpus."""
    news_source_count = defaultdict(int)
    news_source_wcs = defaultdict(list)
    missing_authors = 0

    with closing(sqlite3.connect(DB_FILE_NAME)) as conn:
        curs = conn.cursor()
        for title, authors, pub_date, url, text, tags in get_articles(curs):
            news_source = get_url_base(url)
            news_source_count[news_source] += 1
            news_source_wcs[news_source].append(len(text.split()))
            if authors == '':
                missing_authors += 1

    return news_source_count, news_source_wcs, missing_authors


def show_stats(news_source_count, news_source_wcs, missing_authors):
    """Display statistics on articles."""
    for site, count in sorted(news_source_count.items(), key=itemgetter(1)):
        word_counts = news_source_wcs[site]
        avg_len = round(mean(word_counts), 1)
        std_len = round(stdev(word_counts), 1) if len(word_counts) > 1 else ''
        print(f'{site} has {count} articles with mean length of {avg_len} '
              f'words with stdev {std_len}')

    print(f'\n\nArticles missing author: {missing_authors}')

    word_counts = np.array(list(chain.from_iterable(news_source_wcs.values())))
    sns.kdeplot(word_counts, bw=2)

if __name__ == '__main__':
    article_counts, word_counts, missing_authors = calculate_stats()
    show_stats(article_counts, word_counts, missing_authors)
