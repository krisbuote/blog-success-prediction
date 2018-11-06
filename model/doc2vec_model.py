from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile, common_texts
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np

# # Load and Preprocessing Steps
# Here we load the data and fill in the missing values
filename = 'cleaned_medium_data_Jan_to_August_2018'
text_field = 'text_content'
train = pd.read_csv("../data/" + filename +".csv", usecols=['text_content', 'claps', 'num_words'])
length_before = len(train)
train = train.dropna()
length_after = len(train)
print('{0} / {1} examples were dropped due to NaN'.format(length_before-length_after, length_before))

# ''' TESTING: Drop blogs with 0 claps'''
train = train[train.claps != 0]
print(train.shape[0], 'total sentences.')

# Clip outliers
outlier_cutoff = np.percentile(train['claps'], 95)
c_vec = np.clip(train['claps'], 0, outlier_cutoff)
train['claps'] = c_vec
median_score = np.median(c_vec)
std_score = train['claps'].std()

# Create labels -- transform [claps]
transform_method = 'none'

# claps zscore transform
def transform(df, method):
    '''
    :param method: transformation method [string]. (e.g. 'zscore' or 'log')
    :return: data labels
    '''
    if method == 'none':
        return df['claps']

    elif method == 'zscore':
        print('Outlier cutoff: ', outlier_cutoff, '\nMedian claps: ', median_score, '\nStandard Deviation: ', std_score)
        # df['claps_transform'] = ((c_vec-median_score)/std_score)
        return ((c_vec-median_score)/std_score)

    elif method == 'log':
        # Claps Log transform
        # df['claps_transform'] = np.log(df['claps'])
        return np.log(df['claps'])

    else:
        raise Exception('Wrong data structure or method! See the transform function.')


train['claps_transform'] = transform(train, transform_method)

# Split data into training/testing sets.
train, test = train_test_split(train,
                               random_state = 2018,
                               test_size = 0.25,
                               stratify = pd.qcut(c_vec.values, 10, duplicates = 'drop'))
print('train sentences:', train.shape[0], '. test sentences:', test.shape[0])


# Set the labels as transform of choice (e.g. zscore, log, etc.)
train_y = train['claps_transform']
test_y = test['claps_transform']

# Tokenize the blogs

# ## Sequence Generation
# define network parameters
# max_features = 20000
# Use the average length of articles to pad/truncate all articles to match
# maxlen = int(train['num_words'].max())
# print('Max length of sequences: ', maxlen)


# Create and train doc2vec model
data = train[text_field].values
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 100
vec_size = 100
alpha = 0.025

# Note: dm defines the training algorithm. If dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW).
model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    model.save("./models_weights/d2v.model")

model.save("./models_weights/d2v.model")
print("Model Saved")

# TODO: WHERE TO GO FROM HERE WITH doc2vec MODEL???

# Use doc2vec model to predict similarity of new document
model= Doc2Vec.load("./models_weights/d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])

#
#
# fname = get_tmpfile("my_doc2vec_model")
# model.save(fname)
# model = Doc2Vec.load(fname)  # you can continue training with the loaded model!
#
# # If you've finished training a model, you can save memory with
# # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


