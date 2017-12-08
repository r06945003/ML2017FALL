import sys
import numpy as np
import csv
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Bidirectional
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback
from sklearn import cross_validation, ensemble, preprocessing, metrics
from keras.utils import np_utils
from gensim.models import word2vec
import logging
from keras.preprocessing import sequence

model = word2vec.Word2Vec.load('hw4.word2vec')
word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
for i in range(len(vocab_list)):
	word = vocab_list[i][0]
	word2idx[word] = i + 1
	embeddings_matrix[i + 1] = vocab_list[i][1]

y = []
text = open(sys.argv[1], 'r') 
row = csv.reader(text , delimiter=" ")
for r in row:
    y.append(r[0])
    
text.close()
y = np.array(y)
y = y.astype(np.int)

n_row = 0
x = []
max_length = 0
text = open('training_label_withoutpunc.txt', 'r') 
row = csv.reader(text , delimiter="\n")
for r in row:
    x.append([])
    #for s in r.split():
        #x[n_row].append(r[s])
    tmp = r[0]
    tmp = tmp.split()
    for s in tmp:
        x[n_row].append(s)
    if len(x[n_row]) > max_length:
        max_length = len(x[n_row])
    x[n_row] = np.array(x[n_row])
    n_row = n_row+1 
text.close()
x = np.array(x)
print (max_length)

xx = x.shape[0]
for i in range(xx):
    yy = x[i].shape[0]
    for j in range(yy):
        if x[i][j] in word2idx:
            tmp = word2idx[x[i][j]]
        else:
            tmp = word2idx["_PAD"]
        x[i][j] = tmp
    x[i] = x[i].astype(int)

x = sequence.pad_sequences(x, maxlen=39)

x_train = x[0:180000]
x_val = x[180000:200000]
y_train = y[0:180000]
y_val = y[180000:200000]

model_train = Sequential()
model_train.add(Embedding(65601, 256, input_length=39 ,weights=[embeddings_matrix], trainable=False))
model_train.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False), merge_mode='add'))
#model_train.add(LSTM(64, dropout=0.1, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
#model_train.add(LSTM(32))
#model_train.add(Dense(10, activation='relu'))
#model_train.add(Dropout(0.5))
model_train.add(Dense(1))
model_train.add(Activation("sigmoid"))
model_train.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

filepath = "hw4.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model_train.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=callbacks_list, validation_data=(x_val, y_val))