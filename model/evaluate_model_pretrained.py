import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import load_model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

''' 
This script predicts "claps" on medium articles. The articles have been preprocessed already with 
preprocess_articles.py
'''



# # Load and Preprocessing Steps
# Here we load the data and fill in the missing values
filename = 'cleaned_medium_data_Jan_to_August_2018'
text_field = 'text_content'
train = pd.read_csv("../data/" + filename +".csv")
length_before = len(train)
train = train.dropna()
length_after = len(train)
print('{0} / {1} examples were dropped due to NaN'.format(length_before-length_after, length_before))

# ''' TESTING: Drop blogs with 0 claps'''
train = train[train.claps != 0]
print(train.shape[0], 'total sentences.')

# Clip outliers
outlier_cutoff = np.percentile(train['claps'], 90)
c_vec = np.clip(train['claps'], 0, outlier_cutoff)
train['claps'] = c_vec
median_score = np.median(c_vec)
std_score = train['claps'].std()

# Create labels -- transform [claps]
transform_method = 'zscore'

# claps zscore transform
def transform(df, method):
    '''
    :param method: transformation method [string]. (e.g. 'zscore' or 'log')
    :return: data labels
    '''
    if method == 'zscore':
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
print('train sentences', train.shape[0], 'test sentences', test.shape[0])
train.sample(5)

list_sentences_train = train[text_field].fillna("Invalid").values
list_sentences_test = test[text_field].fillna("Invalid").values

# Set the labels as transform of choice (e.g. zscore, log, etc.)
train_y = train['claps_transform']
test_y = test['claps_transform']

# Plot the training/test distributions
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
ax1.hist(train_y, 10, label = 'Train Scores')
ax1.hist(test_y, 10, label = 'Test Scores', alpha = 0.5)
plt.title('Training/Testing Labels distribution')
ax1.legend()
plt.show()

# ## Sequence Generation
# define network parameters
max_features = 20000
# Use the average length of articles to pad/truncate all articles to match
numWords = [len(x.split()) for x in train['text_content'].values]
maxlen = int(np.mean(numWords))

# Here we take the data and generate sequences from the data
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))

''' This currently only trains on first [maxlen] of words!'''
# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='pre', truncating='post')

# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

model_path = './models_weights/full_model_pretrained.h5'
weights_path = './models_weights/weights_pretrained.hdf5'
model = load_model(model_path)
model.load_weights(weights_path)
model.summary()


batch_size = 64
epochs = 100

# # Make Predictions
# Load the model and make predictions on the test dataset

model.evaluate(X_te, test_y)
predictions = model.predict(X_te)
def inverseTransform(predictions, method):
    if method =='zscore':
        return predictions*std_score+median_score # zscore inverse
    elif method =='log':
        return np.exp(predictions)

test['pred_claps'] = inverseTransform(predictions, transform_method)
fig, ax1 = plt.subplots(1, 1, figsize = (5, 5))
ax1.plot(test['claps'], test['pred_claps'], 'ro', label = 'Prediction')
ax1.plot(test['claps'], test['claps'], 'b-', label = 'Actual')
ax1.legend()
plt.show()

# # Biggest Disappointments

test['pred_error'] = test['claps']-test['pred_claps']
print(test.sort_values('pred_error')[['title', 'claps', 'pred_claps']].head(5))

# surprises
print(test.sort_values('pred_error')[['title', 'claps', 'pred_claps']].tail(5))
