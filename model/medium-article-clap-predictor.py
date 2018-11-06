import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

''' This script predicts "claps" on medium articles. The articles have been preprocessed already with 
preprocess_articles.py
'''


# # Load and Preprocessing Steps
# Here we load the data and fill in the missing values
text_field = 'text_content'
filename = 'cleaned_medium_data_Jan_to_August_2018'
train = pd.read_csv("../data/" + filename +".csv")
length_before = len(train)
train = train.dropna()
length_after = len(train)
print('{0} / {1} examples were dropped due to NaN'.format(length_before-length_after, length_before))

# ''' TESTING: Drop blogs with 0 claps'''
# train = train[train.claps != 0]
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
        df['claps_transform'] = ((c_vec-median_score)/std_score)
        return df['claps_transform']

    elif method == 'log':
        # Claps Log transform
        df['claps_transform'] = np.log(df['claps'])
        return df['claps_transform']
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
maxlen = 100

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

def build_model(conv_layers = 2, max_dilation_rate = 3):
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2*embed_size, 
                   kernel_size = 3)(x)
    prefilt_x = Conv1D(2*embed_size, 
                   kernel_size = 3)(x)
    out_conv = []
    # dilation rate lets us use ngrams and skip grams to process 
    for dilation_rate in range(max_dilation_rate):
        x = prefilt_x
        for i in range(3):
            x = Conv1D(32*2**(i), 
                       kernel_size = 3, 
                       dilation_rate = dilation_rate+1)(x)    
        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    x = concatenate(out_conv, axis = -1)    
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="tanh")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mean_absolute_error'])

    return model

model = build_model()
model.summary()

batch_size = 64
epochs = 100

file_path="./models_weights/weights_pretrained.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks_list = [checkpoint, early] #early
model.fit(X_t, train_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=callbacks_list)

model.load_weights(file_path)
model.save('./models_weights/full_model_pretrained.h5')


# # Make Predictions
# Load the model and make predictions on the test dataset

model.evaluate(X_te, test_y)
predictions = model.predict(X_te)
def inverseTransform(predictions, method):
    if method =='zscore':
        return predictions*std_score+median_score # zscore inverse
    elif method =='log':
        return np.exp(predictions)
    else:
        raise Exception('Wrong data structure or method! See the inverseTransform function.')

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