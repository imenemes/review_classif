from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import pandas as pd
# import the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Classif.models.train_model import y_sent, y_rate, data, X_features_all


# read results
result = pd.read_csv('Sources/results.csv')
print(result.head())

REG_LIN = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=None)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# fonction qui permet la visualisation de la matrice de confusion
def draw_mat(conf_mat,n,names=["negative", "positive"]):
  matrice_proportions = np.zeros((n,n))
  for i in range(0,n):
      matrice_proportions[i,:] = conf_mat[i,:]/float(conf_mat[i,:].sum())
  confusion_df = pd.DataFrame(matrice_proportions, index=names,columns=names)
  plt.figure(figsize=(5,5))
  sns.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='YlGnBu',cbar=False, square=True,fmt='.2f')
  plt.ylabel(r'Valeur réelle',fontsize=14)
  plt.xlabel(r'Valeur prédite',fontsize=14)
  plt.tick_params(labelsize=12)
  plt.show()

liste =[data.Review, X_features_all]

liste_y=[y_sent, y_rate]


def visualise(liste, liste_y):
    for data in liste:
        for y in liste_y:
            CONF_MAT = confusion_matrix(cross_val_predict(REG_LIN, data, y, cv=5), y)
            if len(y.unique()) > 2:
                draw_mat(CONF_MAT, 5, ["1", "2", "3", "4", "5"])
            else:
                draw_mat(CONF_MAT, len(y.unique()))



visualise(liste, liste_y)

