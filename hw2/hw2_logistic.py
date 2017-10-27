import numpy as np  
import pandas as pd  
import csv 
import math
import sys

from sklearn import tree
from sklearn.cross_validation import train_test_split

def sigmoid(z):  
	return 1 / (1 + np.exp(-z))

def predict(x,w,b):
	probability = sigmoid(np.dot(x,w) + b)
	return [1 if h >= 0.5 else 0 for h in probability]

def predict_test(x,w,b):
	probability = sigmoid(np.dot(x,w) + b)
	return [1 if probability >= 0.5 else 0 ]

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


w = np.zeros(106)
b = 0
iteration = 80
lr = 0.0001

#lamda = 1

#s_gra = np.zeros(106)
for i in range(iteration):
    grad = np.zeros(106)
    error = sigmoid(np.dot(x,w) + b) - y
    for j in range(106):
        grad[j] = np.dot(error , x[:,j])
    #s_gra += grad**2
    #ada = np.sqrt(s_gra)
    #w = w - lr * grad/ada
    #grad = grad + lamda * w
    w = w - lr * grad
    b = b - lr * np.sum(error)
    
    pred = np.array(predict(x,w,b)) 
    accur = 1 - np.sum(np.abs(pred - y)) / len(pred)
    print ('iteration: %d | accur: %f  ' % ( i,accur))
    
    
#np.save('hw2_0.85466_0.0001_80.npy',w)

#w = np.load('hw2_0.85466_0.0001_80.npy')


ans = []
for i in range(len(x_test)):
	ans.append([str(i+1)])
	a = predict_test(x_test[i,:],w,b)
	ans[i].append(a[0])

filename = sys.argv[6]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()