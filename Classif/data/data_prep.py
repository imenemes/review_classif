import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



def ratings(rating):
    if rating>3 and rating<=5:
        return 1
    if rating>0 and rating<=3:
        return 0

def word_count(review):
    review_list = review.split()
    return len(review_list)


def clean(review):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    review = re.sub(r'https?://\S+', '', review)  # remove http links
    review = re.sub(r"[^a-zA-Z]", " ", review)  # remove punktuation and numbers

    review = " ".join([word for word in review.split() if word not in stop_words])
    #print(review)
    review = [lemmatizer.lemmatize(review).lower().strip()]  # lematization and stop words removal"""

    return ' '.join(review)

def corpus(text):
    text_list = text.split()
    return text_list


# text prerocessing
def text_preprocess(text):
    """
    Tokenize
    Input: Raw text

    Output: Lemmatized texts
    """

    text = re.sub(r'https?://\S+', '', text)  # remove http links
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove punktuation and numbers

    tokens = word_tokenize(text)  # tokenize to words

    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if
                    tok not in stop_words]  # lematization and stop words removal

    return ' '.join(clean_tokens)