#!/usr/bin/env python3
"""
This module trains supervised learners to predict the validity of news
articles.
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LSHForest
from code.article_db import ArticleDB


def train_model(data: object) -> object:
    """Trains classifier on data and reports test set accuracy."""
    X, y = data.X, data.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lshf = LSHForest(random_state=42)
    lshf.fit(X_train)
    distances, indices = lshf.kneighbors(X_test, n_neighbors=2)
    print(f'{indices} with distance {distances}:')

def article_LSH_neighbors():
    """
    Run against article db to find neighbors of articles.
    """
    articles = ArticleDB(domain_endings=False, author=False)
#    xtrain = articles[6:10, :]

#    train_model(xtrain)

if __name__ == '__main__':
    articles = ArticleDB(domain_endings=False, author=False,
                         source_count=False, misspellings=False, word_count=False, tags=False, tfidf=False, title=True,
                         start_date='2017-03-01',end_date='2017-03-02')
    points_x = articles.X
    points_y = articles.y
    rows = points_x.tocsc()
    print (rows)
    #print(points_x)
    #print(points_y)
