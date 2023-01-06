import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()
data = pd.read_pickle('data_prep.txt')


def add_sentiment_features(data):
    print("adding polarity feature......")
    data['polarity'] = data['Review'].apply(lambda x: (TextBlob(x).sentiment.polarity))
    print("polarity was added")
    print("adding vader feature......")
    data['scores'] = data['Review'].apply(lambda review: sid.polarity_scores(review))
    data['compound'] = data['scores'].apply(lambda score_dict: score_dict['compound'])
    data['neg'] = data['scores'].apply(lambda score_dict: score_dict['neg'])
    data['pos'] = data['scores'].apply(lambda score_dict: score_dict['pos'])
    print("Vader was added")
    return data


data_sent = add_sentiment_features(data)
data_sent.to_pickle('data_sent.txt')
