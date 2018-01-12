import numpy as np
from numpy.linalg import svd, eig
from skimage import io
import sys
import os

X = []
file_pos = []
for file in os.listdir(sys.argv[1]):
    img = io.imread(sys.argv[1] + '/' + file)
    #img.shape
    img = img.flatten()
    X.append(img)
    file_pos.append(file)
X = np.array(X)

X_mean = np.mean(X, axis=0)

#io.imsave('pca_1.jpg', X_mean.reshape(600,600,3).astype(np.uint8))
#plt.imshow(X_mean.reshape(600,600,3).astype(np.uint8))
#plt.title("Average Face")
#plt.show()

X = X - X_mean
X = X.T
#print(X.shape)
y = X[:,file_pos.index(sys.argv[2])]  #which image to reconstruct
#print(y.shape)

U, s, V = np.linalg.svd(X, full_matrices=False)
#U = np.load('U_1.npy')
#s = np.load('s_1.npy')
#V = np.load('V_1.npy')

k = []
for i in range(4):
    k.append(np.dot(y, U[:,i]))

#print(len(k))

re = np.zeros(1080000)
for i in range(4):
    re = re + k[i]*U[:,i]

re += X_mean
re -= np.min(re)
re /= np.max(re)
re = (re * 255)
re = re.astype(np.uint8)


io.imsave('reconstruction.jpg', re.reshape(600,600,3), quality=100)
