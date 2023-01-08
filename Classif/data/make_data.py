import pandas as pd
from Classif.data.make_features import *

df = pd.read_csv('Sources/Data.csv')


def analyse(data):
    data['Sentiment'] = data['Rating'].apply(ratings)
    data['char_count'] = data['Review'].str.len()
    data['Word_count'] = data['Review'].apply(word_count)
    data['Average_Word_Length'] = data['char_count']/data['Word_count']
    data['lem'] = data['Review'].apply(clean)
    data['Review_lists'] = data['lem'].apply(corpus)
    return data


data = analyse(df)

data.to_pickle('Sources/data_prep.txt')



