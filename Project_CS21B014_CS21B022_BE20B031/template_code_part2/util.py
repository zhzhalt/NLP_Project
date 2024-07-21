# Add your import statements here
import json
import argparse
import nltk
import math
from collections import defaultdict
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import words as valid_words



# Add any utility functions here

def stopwords_calculator():
    # obtaining the documents from the cranfield dataset
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-dataset', default="cranfield/", help="Path to the dataset folder")
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print("Warning: Unrecognized command-line arguments:", unknown_args)

    docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:][:]
    docs = [item["body"] for item in docs_json]

    N = len(docs)
    # getting the sentences in the corpus across all documents
    doc_sentences = []
    for doc in docs:
        doc_sentences.extend(tokenize.sent_tokenize(doc))  
    # creating a dictionary of words that has keys as the unique words and values as the number of times that key appears in the corpus 
    words = {}
    for sentence in doc_sentences:
        list_words = tokenize.word_tokenize(sentence)
        for w in list_words:
            if w not in words:
                words[w] = 1
            else:
                words[w] +=1
    # creating a dictionary that has unique tokens as the keys and values as the number of documents the key appears in
    word_count = defaultdict(int)
    for word in words.keys():
        for doc in docs:
            if word in doc:
                word_count[word] += 1
    # creating a dictionary that has unique tokens as the keys and IDF values as the values
    word_idf = defaultdict(int)
    for word in words.keys():
        word_idf[word] = math.log(N/word_count[word])
    return word_idf
    

# finding the nltk stopwords

stop_words = set(stopwords.words('english'))

idf = stopwords_calculator()

common_nltk_stopwords = []
domain_specific_stopwords = []

# finding the nltk stopwords that are also in the corpus of documents and finding domain specific stopwords with threshold 0.8, making sure it is a valid word and not a single letter

letters = 'abcdefghijklmnopqrstuvwxyz'
for key, value in idf.items():
    if key in stop_words:
        common_nltk_stopwords.append(key)
    if (value < 0.8) and key in valid_words.words() and key not in letters:
        domain_specific_stopwords.append(key)

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import seaborn as sns


