import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from nltk.internals import Counter
from tqdm import trange
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt




#read data
data = pd.read_pickle('Sources/data_sent.txt')

plt.pie(data['Sentiment'].value_counts(), labels=data['Sentiment'].unique().tolist(), autopct='%1.1f%%')
plt.title('Le pourcentage des sentiments positif et les sentiments negatif')
plt.show()


graph = sns.FacetGrid(data=data, col='Sentiment')
graph.map(plt.hist, 'Word_count', bins=50)
plt.show()

plt.boxplot(data['Word_count'])
plt.show()

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate\
        (" ".join(data['lem'].astype(str)))
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Le cloud des mots')
plt.show()


features = data.columns.tolist()[2:]
df = data.drop(features, axis=1)
corpus = []
for i in trange(df.shape[0], ncols=150, nrows=10, colour='green', smoothing=0.8):
    corpus += data['Review_lists'][i]
len(corpus)
mostCommon = Counter(corpus).most_common(10)
words = []
freq = []
for word, count in mostCommon:
    words.append(word)
    freq.append(count)

data.head(10)
sns.barplot(x=freq, y=words)
plt.title('Top 10 Most Frequently Occuring Words')
plt.show()

