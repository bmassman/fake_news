#!/usr/bin/env python3
"""
Script to scrape internet news sites for articles.
"""
import sqlite3
import os
import datetime as dt
from contextlib import closing
import newspaper
import pytz

URL_FILE_NAME = 'news_sites.txt'
DB_FILE_NAME = 'articles.db'


def get_configuration():
    """Return configuration for news site scraping."""
    conf = newspaper.Config()
    conf.memoize_articles = False
    conf.fetch_images = False
    conf.MIN_WORD_COUNT = 1
    conf.MAX_TEXT = 6 * 5000
    return conf


def build_news_sources(url_file_name):
    """Return list of built news sites from url_file_name."""
    conf = get_configuration()
    with open(url_file_name, 'r') as f:
        news_urls = f.read().splitlines()
    for news_url in news_urls:
        yield newspaper.build(news_url, config=conf)


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
    authors = ','.join(article.authors)
    publish_date = article.publish_date.isoformat()
    tags = ','.join(article.tags)
    fields = (article.title, authors, publish_date, article.url,
              article.text, tags)
    command = 'INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)'
    curs.execute(command, fields)


def is_in_db(curs, url):
    """Return True if url is already in database, False otherwise."""
    command = ('SELECT * '
               'FROM articles '
               'WHERE url=?')
    curs.execute(command, (url,))
    return curs.fetchone() is not None


def is_new(curs, article):
    """
    Return True if article is new, False otherwise.
    New is defined as not being already present in the database with a
    publish date of 1/Feb/2017 or later.
    """
    pub_date = article.publish_date
    eastern = pytz.timezone('US/Eastern')
    if not pub_date:
        return False
    if not pub_date.tzinfo:
        pub_date = pub_date.astimezone(eastern)
    if pub_date < dt.datetime(2017, 2, 1, tzinfo=eastern):
        return False
    return True


def get_articles(curs):
    news_sites = build_news_sources(URL_FILE_NAME)
    for news_site in news_sites:
        news_site.articles[:] = [article for article in news_site.articles
                                 if not is_in_db(curs, article.url)]
        num_articles = len(news_site.articles)
        print(f'Downloading {num_articles} articles from {news_site.url}')
        for i in reversed(range(len(news_site.articles))):
            article = news_site.articles[i]
            yield article
            del news_site.articles[i]


def scrape_news():
    """Populate database with news articles."""
    inserts = 0
    bad_articles = 0

    with closing(get_db_conn(DB_FILE_NAME)) as conn:
        curs = conn.cursor()
        for article in get_articles(curs):
            try:
                article.download()
                article.parse()
            except newspaper.ArticleException:
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


if __name__ == '__main__':
    scrape_news()
