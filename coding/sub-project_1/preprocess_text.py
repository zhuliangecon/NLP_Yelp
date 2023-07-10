#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:43:06 2023

@author: guochundi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 

import os 
import re
import json 
import nltk # Natural Language Toolkit 
import string
import time as tm 
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from multiprocesspandas import applyparallel

#%% functions 
def json_to_csv(read_path, slc_cols = None):
    """Read jason file from the path and convert to a csv file and save it.
    """
    
    # Read JSON objects line by line and store them in a list
    json_objects = []
    tic= tm.time()
    with open(read_path, 'r') as file:
        for line in file:
            json_objects.append(json.loads(line))
    toc = tm.time()
    print("It takes ", round(toc - tic, 2), "seconds to retrieve data.")

    # Convert the list of JSON objects to a DataFrame
    tic = tm.time()
    data = pd.DataFrame(json_objects)
    toc = tm.time()
    print("It takes ", round(toc - tic, 2), "seconds to convert data into a pandas dataframe.")
     
    # Collect the useful varaibles
    if slc_cols == None:
        slc_cols = list(data.columns)
        
    data = data[slc_cols]
    
    return data

def add_ctg_dummy(data, ctg_lst):
    """Add category dummy to business data.
    """
    
    for ctg in ctg_lst:
        data['is_' + ctg.lower()] = data['categories'].apply(lambda x: 
                                                     1 if x is not None and ctg.capitalize() in x else 0)
    
    return data 
    
def preprocess_text(text):
    """Preprocess text using nltk.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove specical characters and digits
    text = re.sub(f"[{string.punctuation}0-9]", " ", text)
    
    # Tokenize words
    words = nltk.word_tokenize(text)
    
    # Lemmatize words back to the standard form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Remove stopwords, i.e. useless words such as "a", "an" and "the"
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    # Only keep English words 
    english_words = set(nltk.corpus.words.words()) 
    words = [word for word in words if word in english_words]
    
    # Rejoin words back to text string
    text = " ".join(words)
    
    return text

def preprocess_reviews(read_path, save_path):
    """Preprocess review file and use parallel processing to speed up.
    """
    tic = tm.time()
    reviews = pd.read_csv(read_path)
    toc = tm.time()
    print("It takes ", round(toc - tic, 2), "seconds to load data into workspace.")
    print()
    
    # Apply the preprocessing function to the 'text' column of the 'review' DataFrame by parallel processing 
    tic = tm.time()
    reviews['clean_text'] = reviews['text'].apply_parallel(preprocess_text, 
                                                                         num_processes = os.cpu_count(), n_chunks = None)
    toc = tm.time()
    print("It takes ", round(toc - tic, 2), "seconds to preprocess the text in review.")
    reviews['clean_review_length'] = reviews['clean_text'].apply(lambda x: len(re.findall(r'\w+', x)))
    
    reviews = reviews[reviews.clean_review_length > 0]
    reviews = reviews.drop(columns=["text"])
    # Write into a csv file
    reviews.to_csv(save_path, index=False)
    
    return reviews 















    
    
    
    
    
    
    