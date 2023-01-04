import pandas as pd
import re
import nltk

from tqdm import trange
from nltk import tokenize
import sklearn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import plotly.express as px
stop_words = set(stopwords.words('english'))
data = pd.read_csv('Data.csv')


def ratings(rating):
    if rating>3 and rating<=5:
        return 1
    if rating>0 and rating<=3:
        return 0

def word_count(review):
    review_list = review.split()
    return len(review_list)


def clean(review):
    review = review.lower()
    review = re.sub('[^a-z A-Z 0-9-]+', '', review)
    review = " ".join([word for word in review.split() if word not in stop_words])

    return review

def corpus(text):
    text_list = text.split()
    return text_list

data['Sentiment'] = data['Rating'].apply(ratings)
data['char_count'] = data['Review'].str.len()
data['Word_count'] = data['Review'].apply(word_count)
data['Average_Word_Length'] = data['char_count']/data['Word_count']
data['Clean'] = data['Review'].apply(clean)
data['Review_lists'] = data['Clean'].apply(corpus)
print(data.head(10))