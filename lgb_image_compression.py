from matplotlib import pyplot as plt
import random
import math
import numpy as np


######################################################################
# LBG - simulation with toy data
######################################################################

def get_cluster(x, y, K, centroids):
    error = 0
    codewords = []
    colors = ["black", "blue", "green", "yellow", "purple"]
    S = [[] for i in range(K)]
    for _dat in data:
        _dist = [ distance(_dat, i) for i in centroids ]
        error += min(_dist)
        _index = _dist.index(min(_dist))
        codewords.append( colors[_index] )
        S[_index].append(_dat)
    return codewords, S, error

num_of_data = 50
data = [ (random.randint(1, 13), random.randint(1, 13)) for i in range(num_of_data) ]

x = [a[0] for a in data]
y = [a[1] for a in data]

K = 5
while len(centroids) != K:
    centroids = [ ( random.randint(min(x),max(x)), random.randint(min(y),max(y)) ) for i in range(K)]
x_cet = [ i[0] for i in centroids ]
y_cet = [ i[1] for i in centroids ]
plt.scatter(x, y, c="blue")
plt.scatter(x_cet, y_cet, c="red")
plt.show()

error = []
distortion = 9999
k = 0
while True:
    codewords, S, err = get_cluster(x, y, K, centroids)
    error.append(err)
    print("================" * 3)
    print(" k : " + str(k))
    if k == 0:
        distortion = 9999
        print(" error : " + str( error[-1] ) )
    else:
        distortion = (error[-2] - err) / err
        print(" error : " + str(distortion) )
    print("================" * 3)
    plt.scatter(x,y,c=codewords)
    plt.scatter(x_cet, y_cet, c="red")
    plt.show()
    
    if distortion < 0.01:
        break
    else:
        new_centroids = []
        for _cds in S:
            _x = 0
            _y = 0
            _len = len(_cds)
            for _c in _cds:
                _x += _c[0]
                _y += _c[1]
            new_centroids.append( (_x/_len,_y/_len) )

        x_cet = [ i[0] for i in new_centroids ]
        y_cet = [ i[1] for i in new_centroids ]
        
        centroids = new_centroids
        k += 1


######################################################################
# LBG - Image Compression
######################################################################


import cv2
from PIL import Image

img_1 = cv2.imread('original.jpeg')

data = []
for i in range(img_1.shape[0]):
    for j in range(img_1.shape[1]):
        data.append( img_1[i][j].tolist() )


def distance(x, y):
    _res = 0
    for i in range( len(x) ):
        _res += (x[i]-y[i])**2
    return math.sqrt( _res )


def get_cluster(K, centroids, data):
    error = 0
    S = [[] for i in range(K)]
    codeword = []
    for _dat in data:
        _dist = [ distance(_dat, i) for i in centroids ]
        error += min(_dist)
        _index = _dist.index(min(_dist))
        S[_index].append(_dat)
        codeword.append( centroids[_index] )
    return codeword, S, error

def lbg(data, K):
    num_of_data = len(data)
    dim = len(data[0])
    
    centroids = []
    while len(centroids) != K:
        for i in range(K):
            centroids.append( [ random.randint(0,255) for j in range(dim) ] )

    error = []
    distortion = 9999
    k = 0
    while True:
        codeword, S, err = get_cluster(K, centroids, data)
        error.append(err)

        if k == 0:
            distortion = error[-1]
        else:
            distortion = (error[-2] - err) / err


        if distortion < 0.1:
            break
        else:
            new_centroids = []
            for _cds in S:
                _tmp = [0] * dim
                _len = len(_cds)
                if _len != 0:
                    for _c in _cds:
                        for i in range(dim):
                            _tmp[i] += _c[i]
                    for i in range(dim):
                        _tmp[i] = _tmp[i] / _len
                
                    new_centroids.append( _tmp )
                else:
                    new_centroids.append( [ random.randint(0,255) for j in range(dim) ] )

            centroids = new_centroids
            k += 1
    return centroids, S, codeword

cent, S, codeword = lbg(data, 30)
target = img_1.copy()
for i in range( img_1.shape[0] ):
    for j in range( img_1.shape[1] ):
        _dat = target[i][j]
        dist = [ distance(_dat, i) for i in cent ]
        index = dist.index(min(dist))
        target[i][j] = cent[index]

plt.imshow(img_1)
plt.imshow(target)
plt.show()

im = Image.fromarray(target)
im.save("100.jpeg")


######################################################################
# K-menas - Image Compression
######################################################################

import cv2
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

num_of_clusters = 20

#Read the image
image = cv2.imread('original.jpeg')
plt.imshow(image)
plt.show()

#Dimension of the original image
rows = image.shape[0]
cols = image.shape[1]

#Flatten the image
image = image.reshape(rows*cols, 3)

#Implement k-means clustering to form k clusters
kmeans = KMeans(n_clusters=num_of_clusters)
kmeans.fit(image)

#Replace each pixel value with its nearby centroid
compressed_image = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

#Reshape the image to original dimension
compressed_image = compressed_image.reshape(rows, cols, 3)

#Save and display output image
im = Image.fromarray(target)
im.save("compressed_image_{}.png".format(num_of_clusters))

