#!/usr/bin/env python
"""
Script to convert sqlite3 database to csv file.
"""
import sqlite3
import csv
from contextlib import closing

DB_FILE_NAME = 'articles.db'
CSV_FILE_NAME = 'articles.csv'


def convert_db(db_file_name, csv_file_name):
    """Write csv file with all data from db_file_name."""
    with closing(sqlite3.connect(db_file_name)) as conn:
        curs = conn.cursor()
        curs.execute('select * from articles')
        names = tuple(description[0] for description in curs.description)
        data = curs.fetchall()

    with open(csv_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(data)


if __name__ == '__main__':
    convert_db(DB_FILE_NAME, CSV_FILE_NAME)
