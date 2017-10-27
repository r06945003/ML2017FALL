import numpy as np  
import pandas as pd  
import csv 
import math
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn import cross_validation, ensemble, preprocessing, metrics


from keras.models import load_model

x = pd.read_csv(sys.argv[3], encoding = 'big5-tw')
x = np.array(x)
x = x.astype(np.float64)
y = pd.read_csv(sys.argv[4], encoding = 'big5_tw')
y = np.array(y)
y = y.astype(np.float64)
y = y.T
y = y[0]

x_test = pd.read_csv(sys.argv[5], encoding = 'big5-tw')
x_test = np.array(x_test)
x_test = x_test.astype(np.float64)

x_mean = np.mean(np.concatenate((x, x_test), 0), 0)
x_std = np.std(np.concatenate((x, x_test), 0), 0)
for k in range(106):
	x[:,k] = x[:,k] - x_mean[k]
	x[:,k] = x[:,k] / x_std[k]
	x_test[:,k] = x_test[:,k] - x_mean[k]  
	x_test[:,k] = x_test[:,k] / x_std[k]



model_pred = load_model('weights_best_sex01.hdf5')

predict = model_pred.predict(x_test)
predict = np.round(predict)

ans = []
for i in range(len(x_test)):
	ans.append([str(i+1)])
	a = int(predict[i,0])
	ans[i].append(a)

filename = sys.argv[6]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()