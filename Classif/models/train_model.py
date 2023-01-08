import pandas as pd

#import the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#separate data method
from sklearn.model_selection import cross_validate

# make a pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
# initialise model and vectorizers
stand = MinMaxScaler()

# call the default vectorizer
tvec = TfidfVectorizer(stop_words=None)

# pipeline for numerical data
numerical_features=["char_count", "Word_count", "Average_Word_Length"]

# construct the column transfomer
column_transformer = ColumnTransformer(
    [('tfidf', tvec, 'Review'),
    ('standard', stand, numerical_features)],
    remainder='passthrough')


"""function with 3 arguments : 
liste: the list of the tested arguments
X: the input data
Y: our target 
vectorizer: TFIDF vectorizer by default
average_methode : by default binary , will changed in multiclass classification"""


def lmodel_cv(list_model, X, Y, vectorizer=tvec, average_method='binary'):
    results = []
    for model in list_model:
        pipe = Pipeline([
                      ('tfidf', vectorizer),
                      ('classify', model)
                    ])
        cv_results = cross_validate(pipe, X, Y, cv=5)
        res = pd.DataFrame(cv_results)
        results.append(res['test_score'].mean())
    return results


