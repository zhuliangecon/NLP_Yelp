#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:37:27 2023

@author: guochundi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 

import re 
import collections
import time as tm 
import numpy as np 
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#%% define functions 

def tokenize(s):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    word_list = re.findall(r'\w+', s.lower())
    # filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    
    filtered_words = [word for word in word_list]
    
    return filtered_words

def count_ngrams(lines, min_length = 2, max_length = 4):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)
    
# Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1
# Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()
# Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()
    return ngrams

def print_most_frequent(ngrams, num = 10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print(f"----- {num} most common {n}-word phrase -----")
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

def trnsfrm_format(val):
    """Transform image format by replacing the numeric expression of white."""
    if val == 0:
        return 255
    else:
        return val
     
def trnsfrm_mask(file_path):
    """Transform image format into a format that is approriate for mask."""
    
    mask = np.array(Image.open(file_path).convert('L'))
        
    transformed_mask = np.ndarray((mask.shape[0], mask.shape[1]), np.int32)
    
    for i in range(len(mask)):
        transformed_mask[i] = list(map(trnsfrm_format, mask[i]))
    
    return transformed_mask


def print_word_cloud(ngrams, star, num = 20):
    """Print word cloud image plot """
    words = []
    for n in sorted(ngrams):
        for gram, count in ngrams[n].most_common(num):
            s = ' '.join(gram)
            words.append(s)
    color_lst = ['Greys', 'pink', 'Oranges', 'Blues', 'Greens']
    ct_color_lst = ['gray', 'hotpink', 'tomato', 'deepskyblue', 'limegreen']
    transformed_mask = trnsfrm_mask('img/single_star.png')
    
    cloud = WordCloud(width=1440, height= 1080, max_words= 200, colormap =  color_lst[int(star) - 1],
                      mask = transformed_mask, contour_color = ct_color_lst[int(star) - 1], 
                      contour_width = 3, background_color = 'white').generate(' '.join(words))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis('off');
    plt.show()
    print('')
    
def compute_show_frequent(reviews, star, max_len = 3, n_frq1 = 10, n_frq2 = 10):
    """Compute most frequent words and print as word cloud plot"""
    tic = tm.time()
    most_frqt_reviews = count_ngrams(reviews, max_length = max_len)
    toc = tm.time()
    print(f"It takes {round(toc - tic, 2)} seconds to generate most frequent words for {int(star)} star.")

    print_word_cloud(most_frqt_reviews, star, n_frq1)
    print_most_frequent(most_frqt_reviews, num = n_frq2)
    
    return most_frqt_reviews
    
    
