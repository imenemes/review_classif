import pandas as pd
import pickle
from Classif.data.data_prep import *

df = pd.read_csv('Data.csv')


def analyse(data):
    data['Sentiment'] = data['Rating'].apply(ratings)
    data['char_count'] = data['Review'].str.len()
    data['Word_count'] = data['Review'].apply(word_count)
    data['Average_Word_Length'] = data['char_count']/data['Word_count']
    data['lem'] = data['Review'].apply(clean)
    data['Review_lists'] = data['lem'].apply(corpus)
    ##data['lem'] = data['Review'].apply(text_preprocess)
    return data

data = analyse(df)

data.to_pickle('data_prep.txt')



