import os, json
import pandas as pd
import numpy as np

from experiment_settings import event,filter_words
import json
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate,LeaveOneOut,KFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
data_path=os.path.join("..","data",event,"clean.csv")


data=pd.read_csv(data_path)

def get_top_n_words(corpus,n):
    vec = CountVectorizer().fit(corpus)
    bag_of_words=vec.transform(corpus)
    sum_words=bag_of_words.sum(axis=0)
    words_freq=[(word, sum_words[0,idx]) for word,idx in 
                vec.vocabulary_.items()]
    words_freq=sorted(words_freq,key= lambda x:x[1],reverse=True)   
    return words_freq[:n]

'''
common_words=get_top_n_words((data["text"]),20)
for word,freq in common_words:
    print(word,freq)
'''

def community_label_analysis():
    print("community_label") 
    for cluster, g in data.groupby("community_label"):
        print("Cluster is\n",cluster) 
        common_words=get_top_n_words((g["text"]),20)
        for word,freq in common_words:
            print(word,freq)
community_label_analysis()











