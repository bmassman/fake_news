#!/usr/bin/env python3
"""
Script to scrape internet news sites for articles.
"""
import sqlite3
import os
import datetime as dt
from contextlib import closing
import newspaper

URL_FILE_NAME = 'news_sites.txt'
DB_FILE_NAME = 'articles.db'


def build_news_sources(url_file_name):
    """Return list of built news sites from url_file_name."""
    with open(url_file_name) as news_urls:
        return [newspaper.build(news_url) for news_url in news_urls]


def build_article_db(db_file_name):
    """Build database for holding article information return cursor."""
    conn = sqlite3.connect(db_file_name)
    command = ('CREATE TABLE articles '
               '(title, authors, publish_date, url, text, tags)')
    with closing(conn.cursor()) as curs:
        curs.execute(command)
    return conn


def get_db_conn(db_file_name):
    """Return conn to article database."""
    if os.path.isfile(db_file_name):
        return sqlite3.connect(db_file_name)
    else:
        return build_article_db(db_file_name)


def insert_article(curs, article):
    """Insert relevent article fields into db."""
    authors = ''.join(article.authors)
    publish_date = article.publish_date.isoformat()
    tags = ''.join(article.tags)
    fields = (article.title, authors, publish_date, article.url,
              article.text, tags)
    command = 'INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)'
    curs.execute(command, fields)


def is_new(curs, article):
    """
    Return True if article is new, False otherwise.
    New is defined as not being already present in the database with a
    publish date of 1/Feb/2017 or later.
    """
    pub_date = article.publish_date
    # TODO: handle dates with and without tz information
    if pub_date is None or pub_date < dt.datetime(2017, 2, 1):
        return False
    command = ('SELECT * '
               'FROM articles '
               'WHERE url=?')
    curs.execute(command, (article.url,))
    return curs.fetchone() is None


def build_config():
    """Return newspaper configuration."""
    config = newspaper.Config()
    config.memoize_articles = False
    config.fetch_images = False
    return config


config = build_config()

# news_sites = build_news_sources(URL_FILE_NAME)
# newspaper.news_pool.set(news_sites, threads_per_source=2)
# newspaper.news_pool.join()

news_sites = [newspaper.build('http://cnn.com', config=config)]

inserts = 0
bad_articles = 0

with closing(get_db_conn(DB_FILE_NAME)) as conn:
    curs = conn.cursor()
    for news_site in news_sites:
        articles = news_site.articles
        print(f'Starting {len(articles)} downloads from {news_site.url}')
        for article in articles:
            try:
                article.download()
                article.parse()
            except ArticleException:
                bad_articles += 1
            else:
                if is_new(curs, article):
                    insert_article(curs, article)
                    inserts += 1
            if inserts % 50 == 0:
                conn.commit()
    conn.commit()

print(f'{inserts} articles inserted')
print(f'{bad_articles} were not downloaded')

