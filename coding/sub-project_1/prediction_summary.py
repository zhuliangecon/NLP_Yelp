#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:54:36 2023

@author: guochundi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 

import time as tm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
from uszipcode import SearchEngine
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#%% vectorization and prediction 

def get_vocab_mtx(data, f = CountVectorizer, ngram_min = 1, ngram_max = 1):
    """Vecotarization and generate feature matrix."""
    tic = tm.time()
    vocab = f(ngram_range=(ngram_min, ngram_max)).fit(data)
    toc = tm.time()
    print("Vectorize the vocabulary in ", ngram_max, "ngrams takes ", round(toc - tic, 2), "seconds.")
    
    print("The length of the vocabulary in reviews is: ")
    print(len(vocab.vocabulary_))
    
    
    tic = tm.time()
    vocab_mtx = vocab.transform(data) 
    toc = tm.time()
    print("Generate the vocabulary matrix in ", ngram_max, "ngrams takes ", round(toc - tic, 2), "seconds.")
    
    print("Shape of the sparse matrix: ", vocab_mtx.shape)
    # Non-zero occurences:
    print("Non-Zero occurences: ", vocab_mtx.nnz)

    # Density of the Matrix 
    density = (vocab_mtx.nnz/(vocab_mtx.shape[0]*vocab_mtx.shape[1]))*100
    print("Density of the matrix = ",density)
    
    
    return vocab, vocab_mtx

def report_prediction(classifier, cls_nm, x_train, x_test, y_train, y_test):
    """Create classifer, train and get prediction and report results."""
    tic = tm.time()
    classifier.fit(x_train, y_train)
    toc = tm.time()
    print("+" * 78)
    print("It takes", round(toc -tic, 2), "seconds to generate ", cls_nm, " Classifer.")
    
    tic = tm.time()
    prediction = classifier.predict(x_test)
    toc = tm.time()
    print()
    print("It takes", round(toc -tic, 2), "seconds to generate ", cls_nm, " prediction.")
    
    print("*" * 78)
    print("Confusion Matrix for ", cls_nm, " Classifier:")
    print(confusion_matrix(y_test, prediction))
    print()
    print("Score:",round(accuracy_score(y_test,  prediction)*100,2))
    print()
    print("Classification Report:",classification_report(y_test,  prediction))
    
    return prediction 
    
#%% generate summary plot 

# transform mask image to compatiable format 
def transform_format(val):
    """Transform image format by replacing the numeric expression of white."""
    if val == 0:
        return 255
    else:
        return val
    
def transform_mask(file_path, pos):
    """Transform image format into a format that is approriate for mask."""
    if pos:
        mask = np.array(Image.open(file_path).convert('L'))
        
    else:
        mask = np.array(Image.open(file_path))
                        
    transformed_mask = np.ndarray((mask.shape[0], mask.shape[1]), np.int32)
    
    for i in range(len(mask)):
        transformed_mask[i] = list(map(transform_format, mask[i]))
    
    return transformed_mask

# generate plots

def gen_word_cloud(coefs, pos, ft_nm):
    """Genertae word cloud for postive or negative features as a prespecified shape."""
    pos_coefs = np.argwhere(coefs > 0).reshape(-1)
    neg_coefs = np.argwhere(coefs < 0).reshape(-1)
    stop_words = ['good', 'great', 'nice', 'well', 'one', 'two', 'restaurant', 'order', 'ordered', 
                  'let', 'say', 'said', 'get', 'got', 'went', 'come', 'came', 'going', 'go', 'think', 'know',
                  'take', 'took', 'make', 'made', 'see', 'saw', 'eat', 'ate', 'taste', 'tasted', 'would','could', 
                  'really', 'definitely', 'also', 'bit','around']
                       
    if pos :
        stop_words.extend(['never', 'little', 'wait', 'long', 'small'])
        transformed_mask = transform_mask('img/thumbs_up.png', pos)
        cloud = WordCloud(width = 1440, height = 1080, max_words = 80, stopwords = stop_words,
                              mask = transformed_mask, colormap = 'Greens', 
                              contour_color = 'olive', contour_width = 3, 
                              background_color = "white").generate(' '.join(ft_nm[pos_coefs]))
    else:
        stop_words.extend(['like', 'love', 'best'])
        transformed_mask = transform_mask('img/thumbs_down.png', pos)
        cloud = WordCloud(width = 1440, height = 1080, max_words = 80, stopwords = stop_words,
                              mask = transformed_mask, colormap = 'Reds', 
                              contour_color = 'firebrick', contour_width = 3,
                              background_color = "white").generate(' '.join(ft_nm[neg_coefs]))
                                         
    plt.figure(figsize = (16, 16))
    plt.imshow(cloud, interpolation = "bilinear")
    plt.axis('off')
    plt.show()
    print('')
    
def gen_bar_plot(coefs, ft_nm):
    """Generate a bar plot to represent the top positive and negative features."""
    pos_coefs = np.argsort(coefs)[-10:]
    neg_coefs = np.argsort(coefs)[:10]
    combine = np.hstack([neg_coefs, pos_coefs])
    colors = ['red' if i < 0 else 'green' for i in coefs[combine]]
    plt.figure(figsize=(16, 9))
    plt.bar(np.arange(2 * 10), coefs[combine], color = colors)
    
    plt.title('Why the restaurant is rated positively or negatively?', fontsize = 15)
    plt.xticks(np.arange( 0, 2 * 10), ft_nm[combine], rotation = 40, ha = 'right')
    plt.show()
    print('') 
    
    
def key_grams(data, b_id, f = TfidfVectorizer, ngram_min = 3, ngram_max = 3, wrd_cld = False, pos = True):
    """Perform a sentimental analysis to categories features into positive or negative."""
    train = data[data['business_id'] == b_id]
    x_train = train['clean_text']
    y_train = train['review_stars'].apply(lambda x: 'good' if x >= 3.0 else 'bad')
    
    # Words vectorization tranform 
    tic = tm.time()
    vocab = f(ngram_range=(ngram_min, ngram_max)).fit(x_train)
    vocab_mtx = vocab.transform(x_train) 
    toc = tm.time()
    # print("It takes", round(toc -tic, 2), "seconds to finish vectorization.")
    
    # SVM regression 
    tic = tm.time()
    classifier = LinearSVC()
    classifier.fit(vocab_mtx, y_train)
    toc = tm.time()
    print(' ')
    # print("It takes", round(toc -tic, 2), "seconds to finish model fitting.")
    
    coefs = classifier.coef_.ravel()
    ft_nm = np.array(vocab.get_feature_names_out())
    
    # generate word cloud 
    if wrd_cld: 
        gen_word_cloud(coefs, pos, ft_nm)
                                         
    # generate bar plot 
    else :
        gen_bar_plot(coefs, ft_nm)
        
#%% allow searching by category and distance 
def get_nearby_zipcd(zipcd, rds):
    """Find nearby zipcode list within a certain range."""
    
    if rds == 0:
        rds = 1
    
    search = SearchEngine()
    zipcd = search.by_zipcode(zipcd)
    
    if zipcd is None:
        print('Uh oh! The zipcode you just entered  is not an appropriate option. Please check your spelling.')
        return zipcd
    else:
        nrby_zipcds = search.by_coordinates(zipcd.lat, zipcd.lng, radius = rds, returns = 5)
        nrby_zipcds = [zipcd.zipcode for zipcd in nrby_zipcds]
        return nrby_zipcds
    
def slct_restaurants(rst_data, nrby_zipcds, ctg, slct_info):
    """Find a list of restaurants given a list of zipcodes and category."""
    if nrby_zipcds is None:
        return pd.DataFrame()
    else:
        is_ctg = 'is_' + ctg.lower()
            
        if is_ctg in rst_data.columns:
            slct_rst_data = rst_data[(rst_data['postal_code'].isin( nrby_zipcds))].query(is_ctg + '==1')
            slct_rst_data = slct_rst_data[slct_info]
        else:
            ctgs = [x[3:].capitalize() for x in rst_data.columns if x.startswith('is_')]
            ctgs = [x for x in ctgs if x != 'Open']
            ctgs = ', '.join(ctgs)
            print(f"Uh oh! {ctg.capitalize()} is not a available choice. The available restaurant categories are:" )
            print(' ')
            print(ctgs)
            slct_rst_data = pd.DataFrame()
       
        return slct_rst_data


def print_summary(rst_data, rv_data, zipcd, rds, ctg, wc, ps):
    """Genertate restaurant summary report."""
    slct_info = ['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'stars', 'review_count', 'categories']
    nrby_zipcds = get_nearby_zipcd(zipcd, rds)
    slct_rsts = slct_restaurants(rst_data, nrby_zipcds, ctg, slct_info)
    
    if slct_rsts.shape[1] == 0: 
        return
    elif slct_rsts.shape[0] == 0:
        print(' ')
        print(f"Uh oh! There is no {ctg.capitalize()} restaurant in the current radius.")
        print("Give another try by choosing a different category or extend your search radius.")
        
    else: 
        print(' ')
        print(f"Oh! We have found {slct_rsts.shape[0]} available choices for you!")
        print(' ')
        for rst_id in slct_rsts['business_id']:
            slct_rst = slct_rsts[slct_rsts['business_id'] == rst_id]
            print('*'*78)
            print(f"The {ctg.capitalize()} restaurant you are looking up now is: ")
            print(' ')
            for col_nm in slct_rsts.columns:
                if (col_nm in ['name', 'address', 'city', 'state' ]):
                    print(f"{slct_rst[col_nm].to_string(index = False)},")
                elif (col_nm == 'postal_code'):
                    print(slct_rst[col_nm].to_string(index = False))
                elif (col_nm == 'stars'):
                    print(' ')
                    print(f"The current rating for this restaurant is {slct_rst[col_nm].to_string(index = False)}")
                elif (col_nm == 'categories'):
                    pd.set_option('max_colwidth', 800)
                    print(' ')
                    print("The main categories that this restaurant belonging to are:")
                    print(' ')
                    print(slct_rst[col_nm].to_string(index = False))
                    
            print('*'*78)
            key_grams(data = rv_data, b_id = rst_id, wrd_cld = wc, pos = ps)
            

