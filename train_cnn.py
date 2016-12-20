from keras.layers import Input, LSTM, Dense, merge, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout
from keras.models import Model
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pdb
from keras import backend as K

import theano.tensor as T # tensor
from theano import function # function
from keras.engine.topology import Layer
import os
import sys
import jieba
jieba.load_userdict("./science")

input_train = sys.argv[1] # s_label_zfli

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/wordvec/'
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

print('Indexing word vectors.')

embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, '125_vec'))
f = open(os.path.join(GLOVE_DIR, 'old_vec'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print('Processing text dataset')

# good

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id

labels = []  # list of label ids
train_left = []
train_right = []

for line in  file(sys.argv[1]): # train
    line  = line.strip()
    tmp = line
    line = line.split('\1')

    if len(line)<5:
        continue

    label_id = line[0]
    tid = line[1]
    title = line[2]
    tid = line[3]
    title_right = line[4].strip() # need strip at this line
    seg_list = jieba.cut(title) 
    seg_list_right = jieba.cut(title_right) 
    text_left = (' '.join(seg_list)).encode('utf-8','ignore').strip()
    text_right = (' '.join(seg_list_right)).encode('utf-8','ignore').strip()
    #print text_left
    #print text_right
    
    texts.append(text_left)
    texts.append(text_right)

    labels.append(float(label_id))
    train_left.append(text_left)
    train_right.append(text_right)


print('Found %s left.' % len(train_left))
print('Found %s right.' % len(train_right))
print('Found %s labels.' % len(labels))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

sequences_left = tokenizer.texts_to_sequences(train_left)
sequences_right = tokenizer.texts_to_sequences(train_right)
#for item  in sequences_left:
#    print item

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_left = pad_sequences(sequences_left, maxlen=MAX_SEQUENCE_LENGTH,padding='pre', truncating='post')
data_right = pad_sequences(sequences_right, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
labels = np.array(labels)

#labels = to_categorical(np.asarray(labels))

# split the data into a training set and a validation set
indices = np.arange(data_left.shape[0])
np.random.shuffle(indices)

data_left = data_left[indices]
data_right = data_right[indices]

labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data_left.shape[0]) # create val and sp

input_train_left = data_left[:-nb_validation_samples]
input_train_right = data_right[:-nb_validation_samples]

val_left = data_left[-nb_validation_samples:]
val_right = data_right[-nb_validation_samples:]

labels_train = labels[:-nb_validation_samples]
labels_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
#print type(word_index)
#for  item in word_index:
#     print item + '\t' + str(word_index[item])
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
'''
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            trainable=True)
'''

print('Training model.')

# train a 1D convnet with global maxpoolinnb_wordsg
#left model


'''
data_1 = np.random.randint(low = 0, high = 200, size = (500, 140))
data_2 = np.random.randint(low = 0 ,high = 200, size = (500, 140))
labels = np.random.randint(low=0, high=2, size=(500, 1))
#labels = to_categorical(labels, 10) # to one-hot
'''

tweet_a = Input(shape=(MAX_SEQUENCE_LENGTH,))
tweet_b = Input(shape=(MAX_SEQUENCE_LENGTH,))

tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,))

embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=True)(tweet_input)

conv1 = Conv1D(128, 3, activation='tanh')(embedding_layer)
drop_1 = Dropout(0.2)(conv1)
max_1 = MaxPooling1D(3)(drop_1)
conv2 = Conv1D(128, 3, activation='tanh')(max_1)
drop_2 = Dropout(0.2)(conv2)
max_2 = MaxPooling1D(3)(drop_2)
#conv2 = Conv1D(128, 3, activation='tanh')(max_1)
#max_2 = MaxPooling1D(3)(conv2)
out_1 = Flatten()(max_1)
#out_1 = LSTM(128)(max_1)
model_encode = Model(tweet_input, out_1) # 500(examples) * 5888

encoded_a = model_encode(tweet_a)
encoded_b = model_encode(tweet_b)

merged_vector = merge([encoded_a, encoded_b], mode='concat') # good
dense_1 = Dense(128,activation='relu')(merged_vector)
dense_2 = Dense(128,activation='relu')(dense_1)
dense_3 = Dense(128,activation='relu')(dense_2)

predictions = Dense(1, activation='sigmoid')(dense_3)
#predictions = Dense(len(labels_index), activation='softmax')(merged_vector)

model = Model(input=[tweet_a, tweet_b], output=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([input_train_left,input_train_right], labels_train, nb_epoch=5)
json_string = model.to_json()  # json_string = model.get_config()
open('my_model_architecture.json','w').write(json_string)  
model.save_weights('my_model_weights.h5')  

score = model.evaluate([input_train_left,input_train_right], labels_train, verbose=0) 
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate([val_left, val_right], labels_val, verbose=0) 
print('Test score:', score[0])
print('Test accuracy:', score[1])
a = model.predict([val_left,val_right])
pdb.set_trace()
i = a
