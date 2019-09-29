# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:32:14 2017

@author: pfierens
"""
from os import listdir
from os.path import join, isdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import sys
from gram_schmidt import gram_schmidt, cmp_eigen, gramschmidt

mypath      = 'att_faces/'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 40
trnperper   = 6
tstperper   = 4
trnno       = personno*trnperper
tstno       = personno*tstperper

#TRAINING SET
images = np.zeros([trnno,areasize])
person = np.zeros([trnno,1])
imno = 0
per  = 0
#Iterate over image person folders
for dire in onlydirs:
    #Iterate over image files
    for k in range(1,trnperper+1):
        #Pixel matrix
        a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
        #Reshape to vector for insertion in 'images'
        images[imno,:] = np.reshape(a,[1,areasize])
        person[imno,0] = per
        imno += 1
    per += 1

#TEST SET
imagetst  = np.zeros([tstno,areasize])
persontst = np.zeros([tstno,1])
imno = 0
per  = 0
#Iterate over image person folders
for dire in onlydirs:
    #Iterate over last 'trnperper' image files
    for k in range(trnperper,10):
        #Pixel matrix
        a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
        #Reshape to vector for insertion in 'images'
        imagetst[imno,:]  = np.reshape(a,[1,areasize])
        persontst[imno,0] = per
        imno += 1
    per += 1
    
#CARA MEDIA
#Mean for pixel i using 'images' columns
meanimage = np.mean(images,0)
fig, axes = plt.subplots(1,1)
axes.imshow(np.reshape(meanimage,[versize,horsize])*255,cmap='gray')
fig.suptitle('Imagen media')

#resto la media
images  = [images[k,:]-meanimage for k in range(images.shape[0])]
imagetst= [imagetst[k,:]-meanimage for k in range(imagetst.shape[0])]

#PCA
#Eigenvectors displayed horizontally in V
U,S,V = np.linalg.svd(images,full_matrices = False)

A = np.transpose(images)
n, m = A.shape
L = np.dot(images, A)

#A = np.array([[60., 30., 20.], [30., 20., 15.], [20., 15., 12.]])
last_R = np.zeros(A.shape)
eig_vec_L = 1
found_eigen = False
i = 0

while not found_eigen:
    Q, R = gram_schmidt(L)
    L = np.dot(R, Q)
    eig_vec_L = np.dot(eig_vec_L, Q)
    found_eigen = cmp_eigen(last_R, R)
    last_R = R
    i += 1

#eig_val_L, eig_vec_L = np.linalg.eigh(L)
eig_vec_C = np.dot(A, eig_vec_L)
    
print(eig_vec_C)
print(R)
print(i)

for i in range(m):
    eig_vec_C[:,i] /= np.linalg.norm(eig_vec_C[:,i])

# Ordenar autovectores de mayor a menor
#LEFT_RIGHT = 1
#flipped_eig_vec_C = np.flip(eig_vec_C, LEFT_RIGHT)
#for i in range(m):
#    flipped_eig_vec_C[:,i] /= np.linalg.norm(flipped_eig_vec_C[:,i])

#custom_eigen = (np.reshape(flipped_eig_vec_C[:,0],[versize,horsize]))*255
#fig_2, axes_2 = plt.subplots(1,1)
#axes_2.imshow(custom_eigen,cmap='gray')
#fig_2.suptitle('Custom autocara')

#print(flipped_eig_vec_C[0,0])
#print(V[0,0])

# imagescov = np.cov(np.transpose(images))
# w, v = np.linalg.eig(imagescov)
# eigen1cov = (np.reshape(v[0,:],[versize,horsize]))*255
# fig, axes = plt.subplots(1,1)
# axes.imshow(eigen1cov,cmap='gray')
# fig.suptitle('Primera autocara')
#print(imagescov)
#L = np.dot(images, np.transpose(images))
#w, v = np.linalg.eig(L)
#eigen1prima = (np.reshape(np.dot(v[:,0], images),[versize,horsize]))*255
#fig, axes = plt.subplots(1,1)
#axes.imshow(eigen1prima,cmap='gray')
#fig.suptitle('Primera autocara prima')
eigen1c = (np.reshape(eig_vec_C[:,0],[versize,horsize]))*255
fig2, axes2 = plt.subplots(1,1)
axes2.imshow(eigen1c,cmap='gray')
fig2.suptitle('Primera autocara con GS')

#Primera autocara...
# reshape: Gives a new shape to an array without changing its data.
eigen1 = (np.reshape(V[0,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen1,cmap='gray')
fig.suptitle('Primera autocara')

plt.show()

sys.exit(0)

eigen2 = (np.reshape(V[1,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Segunda autocara')

eigen3 = (np.reshape(V[2,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Tercera autocara')

nmax = V.shape[0]
nmax = 100
accs = np.zeros([nmax,1])
for neigen in range(1,nmax):
    #Me quedo sólo con las primeras autocaras
    B = V[0:neigen,:]
    #proyecto
    improy      = np.dot(images,np.transpose(B))
    imtstproy   = np.dot(imagetst,np.transpose(B))
        
    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    accs[neigen] = clf.score(imtstproy,persontst.ravel())
    print('Precisión con {0} autocaras: {1} %\n'.format(neigen,accs[neigen]*100))

fig, axes = plt.subplots(1,1)
axes.semilogy(range(nmax),(1-accs)*100)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Error')

