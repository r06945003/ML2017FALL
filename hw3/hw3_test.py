import sys
import numpy as np   
import csv 
import math
from keras.models import load_model
from keras import backend as k
k.set_learning_phase(1)

'''read test.csv'''
x_test = []
n_row = 0
test = open(sys.argv[1], 'r', encoding='big5') 
row = csv.reader(test , delimiter=",")
for r in row:
	# 第0列沒有資訊
	if n_row != 0:
		r[1] = np.array(r[1].split(' '))
		r[1] = np.reshape(r[1], (1, 48, 48))
		x_test.append(r[1])
	n_row = n_row+1
test.close()
x_test = np.array(x_test)
x_test = x_test.astype(np.float32)
x_test = x_test/255

model_pred1 = load_model('AUG_final_8.hdf5')
label_1 = model_pred1.predict(x_test)
label_1 = np.argmax(label_1, axis=1)
label_1 = label_1.reshape([len(label_1),1])

model_pred2 = load_model('AUG_final_7.hdf5')
label_2 = model_pred2.predict(x_test)
label_2 = np.argmax(label_2, axis=1)
label_2 = label_2.reshape([len(label_2),1])

model_pred3 = load_model('AUG_final_9.hdf5')
label_3 = model_pred3.predict(x_test)
label_3 = np.argmax(label_3, axis=1)
label_3 = label_3.reshape([len(label_3),1])

label = np.concatenate((label_1, label_2, label_3), axis = 1)

ans = []
for i in range(len(x_test)):
    ans.append([str(i)])
    ans[i].append(np.argmax(np.bincount(label[i])))

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()