#!/usr/bin/env python3
"""
Script to produce summary statistics for news articles.
"""
import sqlite3
import re
from contextlib import closing
from collections import defaultdict
from urllib.parse import urlparse
from operator import itemgetter

DB_FILE_NAME = 'articles.db'
DOMAIN_ENDING = re.compile(r'\.[a-z]{2,3}$')
DOMAIN_PREFIX = re.compile(r'^[a-z]+\.')


def get_articles(curs):
    """Return cursor with all articles selected."""
    curs.execute('SELECT * FROM articles')
    return curs.fetchall()


def update_news_source_count(news_source_count, url):
    """Updates count of articles by news source."""
    net_loc = urlparse(url).netloc
    if net_loc.startswith('www.'):
        net_loc = net_loc[4:]
    while DOMAIN_ENDING.search(net_loc):
        net_loc = DOMAIN_ENDING.sub('', net_loc)
    net_loc = DOMAIN_PREFIX.sub('', net_loc)
    news_source_count[net_loc] += 1


def calculate_stats():
    """Print statistics on article corpus."""
    news_source_count = defaultdict(int)
    
    with closing(sqlite3.connect(DB_FILE_NAME)) as conn:
        curs = conn.cursor()
        for title, authors, pub_date, url, text, tags in get_articles(curs):
            update_news_source_count(news_source_count, url)
    
    for site, count in sorted(news_source_count.items(), key=itemgetter(1)):
        print(f'{site} has {count} articles')
