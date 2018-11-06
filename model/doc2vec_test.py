from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



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
print('{0} total blogs. {1} were dropped due to 0 claps.'.format(train.shape[0], length_after - train.shape[0]))

# Clip outliers
outlier_cutoff = np.percentile(train['claps'], 95)
c_vec = np.clip(train['claps'], 0, outlier_cutoff)
train['claps'] = c_vec
median_score = np.median(c_vec)
std_score = train['claps'].std()


# Split data into training/testing sets.
train, test = train_test_split(train,
                               random_state = 2018,
                               test_size = 0.25,
                               stratify = pd.qcut(c_vec.values, 10, duplicates = 'drop'))
print('train blogs:', train.shape[0], '. test blogs:', test.shape[0])


# Use doc2vec model to predict similarity of new document
model= Doc2Vec.load("./models_weights/d2v.model")

#to find the vector of a document which is not in training data
data_train = train[text_field].values
data_test = test[text_field].values

test_blog_idx = 11
test_blog = word_tokenize(data_test[test_blog_idx].lower())
v1 = model.infer_vector(test_blog)
print("V1_infer", v1)

# to find most similar doc using tags. most_similar returns top 10 similar tags with their cosine similarities
# similar_docs = model.docvecs.most_similar('1')
similar_docs = model.docvecs.most_similar([v1])

print('Similar doc', similar_docs)

similar_docs_idx, similar_docs_rating = similar_docs[0][0], similar_docs[0][1]
print('Most similar doc index: ', similar_docs_idx, ', similarity: ', similar_docs_rating)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs[similar_docs_idx])
# print(model.docvecs[0])

claps_train = np.log(train['claps'].values)
claps_test = np.log(test['claps'].values)

# Plot the training/test distributions
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
ax1.hist(claps_train, 10, label = 'Train Scores')
ax1.hist(claps_test, 10, label = 'Test Scores', alpha = 0.5)
plt.title('Training/Testing Labels distribution')
ax1.legend()
plt.show()

# Predict claps to be equal to most similar document
claps_predict = claps_train[int(similar_docs_idx)]
actual_claps = claps_test[test_blog_idx]

# Predict with weighted average of 10 most similar document clapss
similar_docs_claps = 0
for idx, similarity in similar_docs:
    similar_docs_claps += claps_train[int(idx)]
avg_claps_predict = similar_docs_claps / len(similar_docs)



print('Predicted claps from similarity: {0}. Actual claps on blog: {1}'.format(claps_predict, actual_claps))
print('Predicted avg claps from similarity: {0}. Actual claps on blog: {1}'.format(avg_claps_predict, actual_claps))



# LOOP OVER ALL TEST DATA
predicted_claps = []
avg_predicted_claps = []

for blog in data_test:
    blog_vector = model.infer_vector(blog)
    similar_docs = model.docvecs.most_similar([blog_vector])

    # Take claps from Most similar blogs
    similar_docs_idx, similar_docs_rating = similar_docs[0][0], similar_docs[0][1]
    claps_predict = claps_train[int(similar_docs_idx)]
    predicted_claps.append(claps_predict)


    # Take average claps from top 10 similar blogs
    similar_docs_claps = 0
    for idx, similarity in similar_docs:
        similar_docs_claps += claps_train[int(idx)]

    avg_claps_predict = similar_docs_claps / len(similar_docs)
    avg_predicted_claps.append(avg_claps_predict)



fig, ax1 = plt.subplots(1, 1, figsize = (5, 5))
ax1.plot(claps_test, predicted_claps, 'ro', label = 'Prediction')
ax1.plot(claps_test, avg_predicted_claps, 'go', label = 'Avg Prediction')
ax1.plot(claps_test, claps_test, 'bo', label = 'Actual')
ax1.legend()
plt.show()







