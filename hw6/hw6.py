import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
from keras.layers import Input, Dense
from keras.models import Model, load_model
import sys

image = np.load(sys.argv[1])
image = image/255. - 0.5

encoder = load_model('hw6_encoder_dim=64_1226_2.hdf5')
image_re = encoder.predict(image)

kmeans_fit = KMeans(n_clusters = 2).fit(image_re)
cluster_labels = kmeans_fit.labels_

text = open(sys.argv[2], 'r') 
row = csv.reader(text , delimiter="\n")
count = 0
first = []
second = []
for r in row:
    if count > 0:
        r = r[0].split(',')
        first.append(int(r[1]))
        second.append(int(r[2]))
    count = 1
text.close()

ans = []
for i in range(len(first)):
    ans.append([str(i)])
    if cluster_labels[first[i]] == cluster_labels[second[i]]:
        a = str(1)
    else:
        a = str(0)
    ans[i].append(a)

filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['ID','Ans'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

