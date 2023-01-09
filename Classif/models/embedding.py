from gensim.models.word2vec import Word2Vec
from Classif.models.train_model import data
from gensim.models import KeyedVectors
import numpy as np

def encode(msg, model, dim):
  return np.mean([model[word] for word in msg if word in model] or [np.zeros(dim)], axis=0)



# Feed a word2vec with the ingredients
model_w2v = Word2Vec(data.Review_lists, vector_size=300, window=5, min_count=2, workers=8, sg=1)

wv=model_w2v.wv
wv.save("Sources/embedding")

# save the model
w2v = KeyedVectors.load('Sources/embedding')
#w2v.similar_by_word("hotel")
# encode the data
X_w2v = np.array([encode(msg, w2v, 300) for msg in data['Review_lists']])
# save the data encoded
print('save')
np.save('Sources/Xembed', X_w2v)
