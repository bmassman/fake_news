#!/usr/bin/env python3
"""
Script to process article database into cleaned dataset.
"""
import re
import pandas as pd


def fix_separators(s: str) -> str:
    """Return s with commas inserted between fields."""
    upper_than_lower = re.compile(r'([a-z])([A-Z])')
    return upper_than_lower.sub(r'\1,\2', s)


def clean_data(articles: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned dataframe of articles."""
    articles['authors'] = articles['authors'].apply(fix_separators)
    articles['tags'] = articles['tags'].apply(fix_separators)
    return articles

