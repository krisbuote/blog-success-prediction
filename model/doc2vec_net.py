''' This script takes in an n-dimensional vector provided by doc2vec model and predicts claps'''

from gensim.models.doc2vec import Doc2Vec


model= Doc2Vec.load("./models_weights/d2v.model")
print(model.docvecs[5])

train = [vec for vec in model.docvecs]
print(train)
