import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



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
