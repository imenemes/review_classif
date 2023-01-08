# import diffrent classification models based on diffrent techniques
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from train_model import *

from sklearn.metrics import classification_report, confusion_matrix , f1_score, roc_auc_score, precision_score, recall_score

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


#read data
data = pd.read_pickle('Sources/data_sent.txt')

# specify our tow target
y_sent = data['Sentiment']
y_rate = data['Rating']

list_model = [LogisticRegression(max_iter=1000), RandomForestClassifier()] #, DecisionTreeClassifier()#, RandomForestClassifier(), KNeighborsClassifier()]


feature_sce_num = data.drop(['Rating', 'Sentiment','lem', 'Review_lists', 'polarity', 'scores', 'pos', 'neg', 'compound'], axis=1)
"""feature_sce_pol = data.drop(['Rating', 'Sentiment','lem', 'Review_lists', 'scores', 'pos', 'neg', 'compound',
                             'char_count', "Word_count", "Average_Word_Length" ], axis=1)"""
feature_sce_pol_num = data.drop(['Rating', 'Sentiment','lem', 'Review_lists', 'scores', 'pos', 'neg', 'compound'], axis=1)
X_features_all = data.drop(['Rating', 'Sentiment','lem', 'Review_lists', 'scores'], axis=1)


def train(list_model, y_sent):
    res = []
    print("training models on clean data lemmatized")
    res.append(lmodel_cv(list_model, data['lem'], y_sent))
    print("training models on brut data review")
    res.append(lmodel_cv(list_model, data['Review'], y_sent))
    print("training models on brut + numéric features")
    res.append(lmodel_cv(list_model, feature_sce_num, y_sent, vectorizer=column_transformer))
    print("training models on brut + numéric features+ pol")
    res.append(lmodel_cv(list_model, feature_sce_pol_num, y_sent, vectorizer=column_transformer))
    print("training models on all features")
    res.append(lmodel_cv(list_model, X_features_all, y_sent, vectorizer=column_transformer))
    return res


result = train(list_model, y_sent)
df = pd.DataFrame(result, columns=list_model)
df.to_pickle('Sources/results.txt')
print(df)

