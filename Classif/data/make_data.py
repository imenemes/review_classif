import pandas as pd

from analyse_func import *

data = pd.read_csv('Data.csv')
print(data.head(10))

def analyse(data):
    data['Sentiment'] = data['Rating'].apply(ratings)
    data['char_count'] = data['Review'].str.len()
    data['Word_count'] = data['Review'].apply(word_count)
    data['Average_Word_Length'] = data['char_count']/data['Word_count']
    data['Clean'] = data['Review'].apply(clean)
    data['Review_lists'] = data['Clean'].apply(corpus)
    return data

analyse(data)
print(data.head(10))


