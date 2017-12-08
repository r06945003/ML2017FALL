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
from keras.models import load_model
import re, string, timeit

exclude = set(string.punctuation)
model = word2vec.Word2Vec.load('hw4.word2vec')
word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
for i in range(len(vocab_list)):
	word = vocab_list[i][0]
	word2idx[word] = i + 1
	embeddings_matrix[i + 1] = vocab_list[i][1]

n_row = 0
x_test = []
max_length = 0
_iter = 0
text = open(sys.argv[1], 'r') 
row = csv.reader(text , delimiter="\n")
for r in row:
    #print(r)
    if _iter>0:
        #if r ==[]: 
            #continue
        x_test.append([])
        r = r[0].split(',',1)[1]
        r = ''.join(ch for ch in r if ch not in exclude)
        r = r.split(' ')
        for s in r:
            if s != '':
                x_test[n_row].append(s)
        if len(x_test[n_row]) > max_length:
            max_length = len(x_test[n_row])
            #print(n_row)
        x_test[n_row] = np.array(x_test[n_row])
        n_row = n_row+1 
    _iter = 1
    #break
text.close()
x_test = np.array(x_test)
print (max_length)

xx = x_test.shape[0]
for i in range(xx):
    yy = x_test[i].shape[0]
    for j in range(yy):
        if x_test[i][j] in word2idx:
            tmp = word2idx[x_test[i][j]]
        else:
            tmp = word2idx["_PAD"]
        x_test[i][j] = tmp
    x_test[i] = x_test[i].astype(int)

x_test = sequence.pad_sequences(x_test, maxlen=39)
model_pred = load_model('hw4_fuck1130.hdf5')
predict = model_pred.predict(x_test)

ans = []
for i in range(len(x_test)):
    ans.append([str(i)])
    a = int(round(predict[i][0]))
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()