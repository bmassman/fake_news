# -*- coding: utf-8 -*-
"""
@author: Eric

Wordcloud code
Thanks to amueller, author of Word Cloud

"""
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt
from sentiment import normalize_text, tokenize_text, sentiments

class SimpleGroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

class GroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


def plotWordCloud(text, keepNeutral=False):
    
    # Normalize the text
    text = normalize_text(text)
    tokens = tokenize_text(text)
    # Pull positive and negative words
    sentiment = sentiments()
    PosList = set(sentiment.Word[sentiment.Affect == 'positive'])
    NegList = set(sentiment.Word[sentiment.Affect == 'negative'])
    Pos = [token for token in tokens if token in PosList]
    Neg = [token for token in tokens if token in NegList]
    
    # Remove neutral words if keepNeutral==False
    if keepNeutral == False:
        tokens = [token for token in tokens if token in (Pos + Neg)]
        text = ' '.join(tokens)
    
    # Determine colors
    color_to_words = {
            '#00ff00': Pos,
            'red': Neg}
    default_color = 'grey'
    grouped_color_func = GroupedColorFunc(color_to_words, default_color)
    wc = WordCloud().generate(text)
    wc.recolor(color_func=grouped_color_func)
    
    # Plot
    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

