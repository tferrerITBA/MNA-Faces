# -*- coding: utf-8 -*-

from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from gram_schmidt import gram_schmidt, cmp_eigen

mypath      = 'att_faces/'
trained_path= 'eigenfaces_kpca.txt'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = len(onlydirs)
trnperper   = 9
tstperper   = 1
trnno       = personno*trnperper

def image_training_kpca():
    trained = Path(trained_path)
    if(trained.is_file()):
        print('Loaded {}'.format(trained_path))
        return np.loadtxt(trained_path)

    #TRAINING SET
    images = np.zeros([trnno,areasize])
    person = np.zeros([trnno,1])
    imno = 0
    per  = 0
    for dire in onlydirs:
        for k in range(1,trnperper+1):
            a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm')
            images[imno,:] = (np.reshape(a,[1,areasize])-127.5)/127.5
            person[imno,0] = dire.split('s')[1]
            imno += 1
        per += 1
    
    #KERNEL: polinomial de grado degree
    # d == degree
    degree = 2
    K = (np.dot(images,images.T)/trnno+1)**degree
            
    #esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([trnno,trnno])/trnno
    K = K - np.dot(unoM,K) - np.dot(K,unoM) + np.dot(unoM,np.dot(K,unoM))    
    
    A = np.copy(K)
    m,n = A.shape
    last_R = np.zeros(A.shape)
    eig_vec_K = 1
    found_eigen = False
    
    while not found_eigen:
        Q, R = gram_schmidt(A)
        A = np.dot(R, Q)
        eig_vec_K = np.dot(eig_vec_K, Q)
        found_eigen = cmp_eigen(last_R, R)
        last_R = R
    
    for i in range(m):
        eig_vec_K[:,i] /= np.linalg.norm(eig_vec_K[:,i])
    
    for col in range(eig_vec_K.shape[1]):
        eig_vec_K[:,col] = eig_vec_K[:,col]/np.sqrt(R[col, col])

    np.savetxt(trained_path, eig_vec_K, fmt='%s')
    
    return eig_vec_K

def batch_testing_kpca(eigenfaces):
    #TRAINING SET
    images = np.zeros([trnno,areasize])
    person = np.zeros([trnno,1])
    imno = 0
    per  = 0
    for dire in onlydirs:
        for k in range(1,trnperper+1):
            a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm')
            images[imno,:] = (np.reshape(a,[1,areasize])-127.5)/127.5
            person[imno,0] = dire.split('s')[1]
            imno += 1
        per += 1
    
    #TEST SET
    tstno       = personno*tstperper
    imagetst  = np.zeros([tstno,areasize])
    persontst = np.zeros([tstno,1])
    imno = 0
    per  = 0
    for dire in onlydirs:
        for k in range(trnperper+1,11):
            a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm')
            imagetst[imno,:]  = (np.reshape(a,[1,areasize])-127.5)/127.5
            persontst[imno,0] = dire.split('s')[1]
            imno += 1
        per += 1
    
    #KERNEL: polinomial de grado degree
    # d == degree
    degree = 2
    K = (np.dot(images,images.T)/trnno+1)**degree
            
    #esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([trnno,trnno])/trnno
    K = K - np.dot(unoM,K) - np.dot(K,unoM) + np.dot(unoM,np.dot(K,unoM))
    
    #pre-proyección
    improypre   = np.dot(K.T,eigenfaces)
    unoML       = np.ones([tstno,trnno])/trnno
    Ktest       = (np.dot(imagetst,images.T)/trnno+1)**degree
    Ktest       = Ktest - np.dot(unoML,K) - np.dot(Ktest,unoM) + np.dot(unoML,np.dot(K,unoM))
    imtstproypre= np.dot(Ktest,eigenfaces)
    
    nmax = eigenfaces.shape[1]
    nmax = 100
    #Me quedo sólo con las primeras autocaras   
    #proyecto
    improy      = improypre[:,0:nmax]
    imtstproy   = imtstproypre[:,0:nmax]
            
    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    labels = clf.predict(imtstproy)
    print(labels)

def input_testing_kpca(eigenfaces, input_image):
    #TRAINING SET
    images = np.zeros([trnno,areasize])
    person = np.zeros([trnno,1])
    imno = 0
    per  = 0
    for dire in onlydirs:
        for k in range(1,trnperper+1):
            a = plt.imread(mypath + dire + '/{}'.format(k) + '.pgm')
            images[imno,:] = (np.reshape(a,[1,areasize])-127.5)/127.5
            person[imno,0] = dire.split('s')[1]
            imno += 1
        per += 1
    
    #TEST SET
    tstno       = 1
    imagetst  = np.zeros([tstno,areasize])
    imno = 0
    imagetst[imno,:]  = (np.reshape(input_image,[1,areasize])-127.5)/127.5
    
    #KERNEL: polinomial de grado degree
    # d == degree
    degree = 2
    K = (np.dot(images,images.T)/trnno+1)**degree
            
    #esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([trnno,trnno])/trnno
    K = K - np.dot(unoM,K) - np.dot(K,unoM) + np.dot(unoM,np.dot(K,unoM))
    
    #pre-proyección
    improypre   = np.dot(K.T,eigenfaces)
    unoML       = np.ones([tstno,trnno])/trnno
    Ktest       = (np.dot(imagetst,images.T)/trnno+1)**degree
    Ktest       = Ktest - np.dot(unoML,K) - np.dot(Ktest,unoM) + np.dot(unoML,np.dot(K,unoM))
    imtstproypre= np.dot(Ktest,eigenfaces)
    
    nmax = eigenfaces.shape[1]
    nmax = 100
    improy      = improypre[:,0:nmax]
    imtstproy   = imtstproypre[:,0:nmax]
            
    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    labels = clf.predict(imtstproy)

    return labels[0]
