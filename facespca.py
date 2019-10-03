# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:32:14 2017

@author: pfierens
"""
from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from gram_schmidt import gram_schmidt, cmp_eigen

mypath      = 'att_faces/'
trained_path= 'eigenfaces_pca.txt'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = len(onlydirs)
trnperper   = 6
trnno = personno*trnperper

def image_training_pca():
    trained = Path(trained_path)
    if(trained.is_file()):
        print('Loaded {}'.format(trained_path))
        return np.loadtxt(trained_path)

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

    np.savetxt(trained_path, eig_vec_C, fmt = '%s')
    return eig_vec_C
    

def batch_testing_pca(eigenfaces):
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
    #eigenfaces = np.reshape(eigenfaces,[areasize,trnno])
    
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
    
def input_testing_pca(eigenfaces, input_image):
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
    #eigenfaces = np.reshape(eigenfaces,[areasize,trnno])
    
    tstno       = 1
        
    #TEST SET
    imagetst  = np.zeros([tstno,areasize])
    imno = 0
    
    #Pixel matrix
    a = input_image/255.0
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
    #print(labels)
    return labels[0]
