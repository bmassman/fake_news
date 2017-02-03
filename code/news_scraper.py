#/usr/bin/env python3
"""
Script to scrape internet news sites for articles.
"""
import newspaper

file_name = 'news_sites'
with open(file_name) as news_urls:
    news_sites = [newspaper.build(news_url) for news_url in news_urls]

newspaper.news_pool.set(news_sites, threads_per_source=2)
newspaper.news_pool.join()

for site in news_sites:
    for article in site.articles:
        article.parse()

