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
from gram_schmidt import gram_schmidt, cmp_eigen

mypath      = 'att_faces/'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 40
trnperper   = 6
trnno = personno*trnperper

class Training:
    trn_images = 0
    person = 0
    eigenfaces= 0
    
    def __init__(self, trn_images, person, eigenfaces):
        self.trn_images = trn_images
        self.person = person
        self.eigenfaces = eigenfaces

def image_training():
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
            person[imno,0] = dire.split('s')[1]
            imno += 1
        per += 1
        
    #CARA MEDIA
    #Mean for pixel i using 'images' columns
    meanimage = np.mean(images,0)
    #fig, axes = plt.subplots(1,1)
    #axes.imshow(np.reshape(meanimage,[versize,horsize])*255,cmap='gray')
    #fig.suptitle('Imagen media')
    
    #resto la media
    images  = [images[k,:]-meanimage for k in range(images.shape[0])]
    
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
    
    for i in range(m):
        eig_vec_C[:,i] /= np.linalg.norm(eig_vec_C[:,i])
    
    return Training(images, person, eig_vec_C)
    

def batch_testing_pca(training):
    #USE TRAINING FIELD!!!!!!!!!!!!
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
            person[imno,0] = dire.split('s')[1]
            imno += 1
        per += 1
    eigenfaces = np.reshape(training.eigenfaces,[areasize,trnno])
    
    tstperper   = 4
    tstno       = personno*tstperper
    
    #TEST SET
    imagetst  = np.zeros([tstno,areasize])
    imno = 0
    #Iterate over image person folders
    for dire in onlydirs:
        #Iterate over last 'trnperper' image files
        for k in range(trnperper+1,11):
            #Pixel matrix
            a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
            #Reshape to vector for insertion in 'images'
            imagetst[imno,:]  = np.reshape(a,[1,areasize])
            imno += 1
        
    #CARA MEDIA
    #Mean for pixel i using 'images' columns
    meanimage = np.mean(images,0)
    #fig, axes = plt.subplots(1,1)
    #axes.imshow(np.reshape(meanimage,[versize,horsize])*255,cmap='gray')
    #fig.suptitle('Imagen media')
    
    #resto la media
    imagetst= [imagetst[k,:]-meanimage for k in range(imagetst.shape[0])]
    
    #PCA
    #Eigenvectors displayed horizontally in V
    #U,S,V = np.linalg.svd(images,full_matrices = False)
    
    
    
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
    #eigen1c = (np.reshape(eig_vec_C[:,0],[versize,horsize]))*255
    #fig2, axes2 = plt.subplots(1,1)
    #axes2.imshow(eigen1c,cmap='gray')
    #fig2.suptitle('Primera autocara con GS')
    
    #Primera autocara...
    # reshape: Gives a new shape to an array without changing its data.
    #eigen1 = (np.reshape(V[0,:],[versize,horsize]))*255
    #fig, axes = plt.subplots(1,1)
    #axes.imshow(eigen1,cmap='gray')
    #fig.suptitle('Primera autocara')
    
    
    #eigen2 = (np.reshape(V[1,:],[versize,horsize]))*255
    #fig, axes = plt.subplots(1,1)
    #axes.imshow(eigen2,cmap='gray')
    #fig.suptitle('Segunda autocara')
    
    #eigen3 = (np.reshape(V[2,:],[versize,horsize]))*255
    #fig, axes = plt.subplots(1,1)
    #axes.imshow(eigen2,cmap='gray')
    #fig.suptitle('Tercera autocara')
    
    #nmax = V.shape[0]
    nmax = 100
    #accs = np.zeros([nmax,1])
    labels = np.zeros([tstno])
    
    neigen = nmax
    #for neigen in range(1,nmax):
        #Me quedo sólo con las primeras autocaras
    B = eigenfaces[:,0:neigen]
    #proyecto
    improy      = np.dot(images,B)
    imtstproy   = np.dot(imagetst,B)
            
        #SVM
        #entreno
    clf = svm.LinearSVC()
    clf.fit(improy, person.ravel())
    labels = clf.predict(imtstproy)
        #print('Precisión con {0} autocaras: {1} %\n'.format(neigen,accs[neigen]*100))
    print(labels)
    #fig, axes = plt.subplots(1,1)
    #axes.semilogy(range(nmax),(1-accs)*100)
    #axes.set_xlabel('No. autocaras')
    #axes.grid(which='Both')
    #fig.suptitle('Error')
    
def input_testing_pca(training, input_image):
    #USE TRAINING FIELD!!!!!!!!!!!!
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
            person[imno,0] = dire.split('s')[1]
            imno += 1
        per += 1
    eigenfaces = np.reshape(training.eigenfaces,[areasize,trnno])
    
    tstno       = 1
        
    #TEST SET
    imagetst  = np.zeros([tstno,areasize])
    imno = 0
    
    #Pixel matrix
    a = input_image/255.0
    #a = plt.imread('detected_faces/10.pgm')/255.0
    #Reshape to vector for insertion in 'images'
    imagetst[imno,:]  = np.reshape(a,[1,areasize])
    #fig, axes = plt.subplots(1,1)
    #axes.imshow(np.reshape(imagetst[0,:],[versize,horsize])*255,cmap='gray')
        
    #CARA MEDIA
    #Mean for pixel i using 'images' columns
    meanimage = np.mean(images,0)
    #fig, axes = plt.subplots(1,1)
    #axes.imshow(np.reshape(meanimage,[versize,horsize])*255,cmap='gray')
    #fig.suptitle('Imagen media')
    
    #resto la media
    imagetst= [imagetst[k,:]-meanimage for k in range(imagetst.shape[0])]
    
    nmax = 100
    #accs = np.zeros([nmax,1])
    labels = np.zeros([tstno])
    
    neigen = nmax
    #for neigen in range(1,nmax):
        #Me quedo sólo con las primeras autocaras
    B = eigenfaces[:,0:neigen]
    #proyecto
    improy      = np.dot(images,B)
    imtstproy   = np.dot(imagetst,B)
            
        #SVM
        #entreno
    clf = svm.LinearSVC()
    clf.fit(improy, person.ravel())
    labels = clf.predict(imtstproy)
        #print('Precisión con {0} autocaras: {1} %\n'.format(neigen,accs[neigen]*100))
    print(labels)
