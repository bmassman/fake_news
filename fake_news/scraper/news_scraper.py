#!/usr/bin/env python3
"""
Script to scrape internet news sites for articles.
"""
import os
import sqlite3
import datetime as dt
from typing import Iterator, Set
from contextlib import closing
from operator import itemgetter
import csv
import newspaper
import pytz
from .article_language import detect_language

URL_FILE_NAME = os.path.join('fake_news', 'scraper', 'news_sites.txt')
DB_FILE_NAME = os.path.join('fake_news', 'articles.db')
GROUND_TRUTH_FILE_NAME = os.path.join('fake_news', 'scraper',
                                      'ground_truth_articles.csv')


def get_configuration() -> newspaper.Config:
    """Return configuration for news site scraping."""
    conf = newspaper.Config()
    conf.memoize_articles = False
    conf.fetch_images = False
    conf.MIN_WORD_COUNT = 1
    conf.MAX_TEXT = 6 * 5000
    return conf


def build_sources(url_file_name: str) -> Iterator[newspaper.Source]:
    """Return list of built news sites from url_file_name."""
    conf = get_configuration()
    with open(url_file_name, 'r') as f:
        news_urls = f.read().splitlines()
    for news_url in news_urls:
        yield newspaper.build(news_url, config=conf)


def build_article_db(db_file_name: str) -> sqlite3.Connection:
    """Build database for holding article information return cursor."""
    conn = sqlite3.connect(db_file_name)
    command = ('CREATE TABLE IF NOT EXISTS articles '
               '(title, authors, publish_date, url, text, tags);'
               'CREATE TABLE IF NOT EXISTS bad_articles (url);'
               'CREATE TABLE IF NOT EXISTS old_articles (url);'
               'CREATE TABLE IF NOT EXISTS ground_truth '
               '(title, authors, publish_date, url, text, tags, labels);')
    with closing(conn.cursor()) as curs:
        curs.executescript(command)
    return conn


def insert_article(curs: sqlite3.Cursor,
                   article: newspaper.Article,
                   table: str) -> None:
    """Insert relevant article fields into db table."""
    if table == 'articles' or table == 'ground_truth':
        authors = ','.join(article.authors)
        try:
            publish_date = article.publish_date.isoformat()
        except AttributeError:
            publish_date = ''
        tags = ','.join(article.tags)
        fields = (article.title, authors, publish_date, article.url,
                  article.text, tags)
        if table == 'articles':
            command = 'INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)'
        else:
            fields += (article.label, )
            command = 'INSERT INTO ground_truth VALUES (?, ?, ?, ?, ?, ?, ?)'
        curs.execute(command, fields)
    elif table == 'bad_articles':
        curs.execute('INSERT INTO bad_articles VALUES (?)', (article.url, ))
    elif table == 'old_articles':
        curs.execute('INSERT INTO old_articles VALUES (?)', (article.url, ))


def is_new(article: newspaper.Article) -> bool:
    """
    Return True if article is new, False otherwise.
    New is defined as a publish date of 1/Feb/2017 or later.
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


def get_previous_urls(curs: sqlite3.Cursor) -> Set[str]:
    """Return set of previously downloaded, attempted or old urls."""
    curs.execute('SELECT url FROM articles')
    urls = set(map(itemgetter(0), curs.fetchall()))
    curs.execute('SELECT url FROM bad_articles')
    urls.update(map(itemgetter(0), curs.fetchall()))
    curs.execute('SELECT url FROM old_articles')
    urls.update(map(itemgetter(0), curs.fetchall()))
    return urls


def get_articles(curs: sqlite3.Cursor) -> Iterator[newspaper.Article]:
    """Yield articles from news sites and ground truth."""
    curs.execute('SELECT url FROM ground_truth')
    previous_urls = set(map(itemgetter(0), curs.fetchall()))
    with open(GROUND_TRUTH_FILE_NAME, 'r') as f:
        reader = csv.reader(f)
        for url, label in reader:
            if url in previous_urls:
                continue
            print(f'Downloading {url}')
            article = newspaper.Article(url)
            article.label = label
            yield article
            previous_urls.add(url)
    news_sites = build_sources(URL_FILE_NAME)
    previous_urls = get_previous_urls(curs)
    for news_site in news_sites:
        news_site.articles[:] = [article for article in news_site.articles
                                 if article.url not in previous_urls]
        num_articles = len(news_site.articles)
        print(f'Downloading {num_articles} articles from {news_site.url}')
        for i in reversed(range(len(news_site.articles))):
            article = news_site.articles[i]
            previous_urls.add(article.url)
            yield article
            del news_site.articles[i]


def scrape_news() -> None:
    """Populate database with news articles."""
    inserts = 0
    ground_truth_inserts = 0
    bad_articles = 0
    old_articles = 0
    foreign_articles = 0

    with closing(build_article_db(DB_FILE_NAME)) as conn:
        curs = conn.cursor()
        for article in get_articles(curs):
            try:
                article.download()
                article.parse()
            except newspaper.ArticleException:
                bad_articles += 1
                insert_article(curs, article, 'bad_articles')
            else:
                if hasattr(article, 'label'):
                    insert_article(curs, article, 'ground_truth')
                    ground_truth_inserts += 1
                elif detect_language(article.text) != 'english':
                    insert_article(curs, article, 'bad_articles')
                    foreign_articles += 1
                elif is_new(article):
                    insert_article(curs, article, 'articles')
                    inserts += 1
                else:
                    old_articles += 1
                    insert_article(curs, article, 'old_articles')
            if inserts % 50 == 0:
                conn.commit()
        conn.commit()

    print(f'{inserts} articles inserted')
    print(f'{ground_truth_inserts} ground truth articles inserted')
    print(f'{bad_articles} were not downloaded')
    print(f'{old_articles} were too old for database')
    print(f'{foreign_articles} were foreign language')


if __name__ == '__main__':
    scrape_news()
