#!/usr/bin/env python3
"""
Module to build dataframe of news articles from sqlite3 database.
"""
import os
import re
from urllib.parse import urlparse
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Optional
import pandas as pd

DB_FILE_NAME = os.path.join('fake_news', 'articles.db')

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


def build_df(table: str = 'articles',
             start_date: Optional[datetime] = None,
             end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Build dataframe with derived fields."""
    with closing(sqlite3.connect(DB_FILE_NAME)) as conn:
        articles = pd.read_sql_query(f'select * from {table}', conn)
    articles['date'] = pd.to_datetime(articles['publish_date'])
    if start_date:
        articles = articles.loc[articles['date'] >= start_date]
    if end_date:
        articles = articles.loc[articles['date'] <= end_date]

    articles = articles.replace([None], [''], regex=True)
    articles['base_url'] = articles.apply(get_url_base, axis=1)
    articles['word_count'] = articles.apply(count_words, axis=1)
    return articles
