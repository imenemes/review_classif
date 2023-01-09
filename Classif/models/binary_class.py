# import diffrent classification models based on diffrent techniques
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from Classif.models.train_model import *



result = train_binary(list_model, liste_feature, y_sent)
# transform result to dataframe
df = pd.DataFrame(result, columns=list_model)
# store it in csv file
df.to_csv('Sources/results_binary.csv', index=False)
print(df)

