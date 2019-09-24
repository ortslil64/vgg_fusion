# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:37:38 2019

@author: ortsl
"""

import numpy as np
import pandas as pd
import cv2
import keras.utils.np_utils as kutils
from keras import backend as k
import tensorflow as tf

def get_fashion_mnist(size = 32):
    train_data = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
    test_data = pd.read_csv('fashionmnist/fashion-mnist_test.csv')

    Y_train = train_data['label']
    Y_test = test_data['label']


    Y_train_ind = np.zeros((len(Y_train),len(set(Y_train))))
    for i in range(len(Y_train)):
        Y_train_ind[i,Y_train[i]] = 1
    
    Y_test_ind = np.zeros((len(Y_test),len(set(Y_test))))
    for i in range(len(Y_test)):
        Y_test_ind[i,Y_test[i]] = 1
        
    
    X_train = train_data.iloc[:,1:].values
    X_test = test_data.iloc[:,1:].values
    
    rgb_list = []
    
    for i in range(len(X_train)):
        img = X_train[i].reshape(28,28)
        img = img.astype(np.float32)
        img = cv2.resize(img,(size,size))
        rgb = np.dstack((img,img,img))
        rgb_list.append(rgb)
        
    rgb_arr = np.stack([rgb_list],axis=4)    
    rgb_arr_to_3d_train = np.squeeze(rgb_arr, axis=4)
    
    rgb_list = []
    
    for i in range(len(X_test)):
        img = X_test[i].reshape(28,28)
        img = img.astype(np.float32)
        img = cv2.resize(img,(size,size))
        rgb = np.dstack((img,img,img))
        rgb_list.append(rgb)
        
    rgb_arr = np.stack([rgb_list],axis=4)    
    rgb_arr_to_3d_test = np.squeeze(rgb_arr, axis=4)
    
    return rgb_arr_to_3d_train,Y_train_ind,rgb_arr_to_3d_test,Y_test_ind


def get_fashion_mnist_1d():
    train_data = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
    test_data = pd.read_csv('fashionmnist/fashion-mnist_test.csv')

    Y_train = train_data['label']
    Y_test = test_data['label']


    Y_train_ind = np.zeros((len(Y_train),len(set(Y_train))))
    for i in range(len(Y_train)):
        Y_train_ind[i,Y_train[i]] = 1
    
    Y_test_ind = np.zeros((len(Y_test),len(set(Y_test))))
    for i in range(len(Y_test)):
        Y_test_ind[i,Y_test[i]] = 1
        
    
    X_train = train_data.iloc[:,1:].values
    X_test = test_data.iloc[:,1:].values
    X_train_shape = (len(X_train),28,28,1)
    X_test_shape = (len(X_test),28,28,1)

    Xtrain = np.empty(X_train_shape)
    Xtest = np.empty(X_test_shape)
    
    for i in range(len(X_train)):
        Xtrain[i,:,:,0] = X_train[i].reshape(28,28) / 255.0
    for i in range(len(X_test)):
        Xtest[i,:,:,0] = X_test[i].reshape(28,28) /255.0
    
    return Xtrain,Y_train_ind,Xtest,Y_test_ind

def fuze_ann(p1,p2,alpha):
    p12 = np.multiply(np.power(p1,alpha),np.power(p2,1-alpha))
    n = np.apply_along_axis(np.sum, 1, p12)
    p12 = p12/n[:,None]
    return p12


def fuze_cnn(probs):
    Nm,D, K = probs.shape
    output = np.empty((D,K))
    for t in range(D):
        A = np.ones(Nm)/Nm
        #A = np.random.random(Nm)
        #A = A/np.sum(A)
        for jj in range(50):
            Q = np.ones(K)
            for ii in range(Nm):
                Q = np.multiply(Q,np.power(probs[ii,t,:],A[ii]))
            idmx = np.argmax(Q)
            for ii in range(Nm):
                A[ii] =  probs[ii,t,idmx]/np.sum(probs[:,t,idmx])
        output[t,:] = Q
    return output

def fuze_cnn_linear(probs):
    Nm,D, K = probs.shape
    output = np.empty((D,K))
    for t in range(D):
        A = np.ones(Nm)/Nm
        #A = np.random.random(Nm)
        #A = A/np.sum(A)
        for jj in range(50):
            Q = np.ones(K)
            for ii in range(Nm):
                Q = np.multiply(Q,np.multiply(probs[ii,t,:],A[ii]))
            idmx = np.argmax(Q)
            for ii in range(Nm):
                A[ii] =  probs[ii,t,idmx]/np.sum(probs[:,t,idmx])
        output[t,:] = Q
    return output

def fuze_cnn_linear(probs):
    Nm,D, K = probs.shape
    output = np.empty((D,K))
    for t in range(D):
        A = np.ones(Nm)/Nm
        #A = np.random.random(Nm)
        #A = A/np.sum(A)
        for jj in range(50):
            Q = np.ones(K)
            for ii in range(Nm):
                Q = np.multiply(Q,np.multiply(probs[ii,t,:],A[ii]))
            idmx = np.argmax(Q)
            for ii in range(Nm):
                A[ii] =  probs[ii,t,idmx]/np.sum(probs[:,t,idmx])
        output[t,:] = Q
    return output
        



def add_noise(data, model,sess,batch_sz = 100, eps = 0.1):
    pb = tf.keras.utils.Progbar(data.shape[0])
    distorted_data = []
    gradients = k.gradients(model.output, model.input)
    for jj in range(data.shape[0]//batch_sz):
        evaluated_gradients = sess.run(gradients, feed_dict={model.input: data[jj*batch_sz:(jj+1)*batch_sz]})
        e = np.sign(evaluated_gradients[0])
        distorted_data.append(data[jj*batch_sz:(jj+1)*batch_sz] + eps*e)
        pb.update(jj*batch_sz)
    return np.array(distorted_data).reshape(data.shape)

























