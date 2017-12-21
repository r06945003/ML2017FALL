import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import csv
import numpy as np
import keras
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, add, Input, Add, Dot
from keras.models import Sequential, Model, load_model
import sys
from keras import backend as K

def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )

text = open(sys.argv[1], 'r') 
row = csv.reader(text , delimiter="\n")
count = 0
user_pred = []
movie_pred = []
for r in row:
    if count > 0:
        r = r[0].split(',')
        user_pred.append(r[1])
        movie_pred.append(r[2])
    count = 1
text.close()
print(len(user_pred))

user_pred = np.array(user_pred)
movie_pred = np.array(movie_pred)

user_pred=user_pred.astype(np.int)
movie_pred=movie_pred.astype(np.int)

max_user_pred = np.max(user_pred)
max_movie_pred = np.max(movie_pred)
print(max_user_pred)
print(max_movie_pred)

user_pred = user_pred - 1
movie_pred = movie_pred - 1

model_pred = load_model('hw5_bias_fuck1210.hdf5', custom_objects={'rmse': rmse})
predict = model_pred.predict([user_pred, movie_pred])
predict = np.clip(predict, 1, 5)

ans = []
for i in range(len(predict)):
    ans.append([str(i+1)])
    #a = float(round(predict[i][0]))
    a = float(predict[i][0])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['TestDataID','Rating'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
