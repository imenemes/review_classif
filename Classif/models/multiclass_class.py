import pandas as pd
from Classif.models.train_model import data, y_rate, list_model, X_features_all, lmodel_cv, column_transformer

from sklearn.model_selection import cross_validate
import numpy as np

X_w2vec = np.load('Sources/Xembed.npy')


def lmodel_cv_embedding(liste, X, Y,average_method='macro'):
  results = []
  for model in liste:
    cv_results = cross_validate(model, X, Y,cv=5)
    res = pd.DataFrame(cv_results)
    results.append(res['test_score'].mean())
  return results



def train_multiclass(list_model, y_rate):
    res = []
    print('training models on brut data')
    res.append(lmodel_cv(list_model, data['Review'], y_rate,average_method='macro'))
    print('training models on all features axtracted')
    res.append(lmodel_cv(list_model, X_features_all, y_rate, vectorizer=column_transformer,average_method='macro'))
    print('training models on brut data vectorized with w2vec')
    res.append(lmodel_cv_embedding(list_model,X_w2v, y_rate,average_method='macro'))
    return res



result = train_multiclass(list_model, y_rate)
# transform result to dataframe
df = pd.DataFrame(result, columns=list_model)
# store it in csv file
df.to_csv('Sources/results_multiclass.csv', index=False)


