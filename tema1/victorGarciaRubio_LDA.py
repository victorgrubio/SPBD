#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:54:55 2019

@author: victor
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
   plt.close('all')
   
   # Select mode changing following variable to this values:
   # 'example': LDA on 3 classes example adapted from 2 classes code
   # 'iris': LDA on iris dataset as from slides
   mode = 'iris'
   
   # Set to True if you want to solve the classification problem
   classify = True
   
   # Example data processing
   if mode == 'example':
      xc1 = np.matrix([[4,2],[2,4],[2,3],[3,6],[4,4]],dtype=np.int16).T
      xc2 = np.matrix([[9,10],[6,8],[9,5],[8,7],[10,8]],dtype=np.int16).T
      xc3 = np.matrix([[14,12],[13,12],[13,14],[13,13],[12,14]],dtype=np.int16).T
      x  = np.concatenate((xc1,xc2,xc3),axis=1)
      plt.figure(1)
      markerline1, _ , _ = plt.stem(np.asarray(xc1[0,:]).flatten(),
                                    np.asarray(xc1[1,:]).flatten(),':')
      markerline2, _ , _ = plt.stem(np.asarray(xc2[0,:]).flatten(),
                                    np.asarray(xc2[1,:]).flatten(),':')
      markerline3, _ , _ = plt.stem(np.asarray(xc3[0,:]).flatten(),
                                    np.asarray(xc3[1,:]).flatten(),':')
      plt.setp(markerline1, 'color', 'b')
      plt.setp(markerline2, 'color', 'm')
      plt.setp(markerline3, 'color', 'r')
      
   # Iris dataset dataprocessing
   elif mode == 'iris':
      data = np.genfromtxt('iris_data.csv', delimiter=',')
      xc1 = np.matrix(data[:50,:4]).T
      xc2 = np.matrix(data[50:100,:4]).T
      xc3 = np.matrix(data[100:150,:4]).T
      x = np.matrix(data[:,:4]).T
      del data
      
   n1 = xc1.shape[1]
   n2 = xc2.shape[1]
   n3 = xc3.shape[1]
   
   mu1 = np.matrix(np.mean(xc1,1),dtype=np.float16)
   mu2 = np.matrix(np.mean(xc2,1),dtype=np.float16)
   mu3 = np.matrix(np.mean(xc3,1),dtype=np.float16)
   mu  = np.matrix(np.mean(x,1),dtype=np.float16)
   # Covariance matrix of the classes
   S1 = np.cov(xc1)
   S2 = np.cov(xc2)
   S3 = np.cov(xc3)
   # Within-class scatter matrix
   Sw = S1 + S2 + S3
   # Between-class scatter matrix
   SB1 = n1*((mu1-mu)*((mu1-mu).T))
   SB2 = n2*((mu2-mu)*((mu2-mu).T))
   SB3 = n3*((mu3-mu)*((mu3-mu).T))
   SB = SB1 + SB2 + SB3
   # Computing the LDA projection
   invSw = np.linalg.pinv(Sw)
   invSwSB = invSw*SB
   # getting the projection vector
   [D,V] = np.linalg.eig(invSwSB)
   # The projection vector
   K=2
   W = np.matrix(np.zeros((V.shape[0],K)))
   for i in range(K):
      W[:,i] = V[:,i]
      
   # Clean unused variables
   del SB,SB1,SB2,SB3,Sw,S1,S2,S3,invSw,invSwSB
   
   # Projection of the classes onto the 2-first vectors LDA
   Yc1=W.T*xc1 
   Yc2=W.T*xc2
   Yc3=W.T*xc3
   Yc1_mu = np.matrix(np.mean(Yc1,1))
   Yc2_mu = np.matrix(np.mean(Yc2,1))
   Yc3_mu = np.matrix(np.mean(Yc3,1))
   Yc_mu = (Yc1_mu+Yc2_mu+Yc3_mu)/3
   # Plotting original data
   plt.figure(2)
   plt.subplot(3,1,1)
   plt.scatter(np.asarray(xc1[0,:]),np.asarray(xc1[1,:]),color='r',marker='o')
   plt.scatter(np.asarray(xc2[0,:]),np.asarray(xc2[1,:]),color='b',marker='^')
   plt.scatter(np.asarray(xc3[0,:]),np.asarray(xc3[1,:]),color='m',marker='+')
   plt.plot(mu1[0],mu1[1],color='r',marker='.',markersize=20,markeredgecolor='r')
   plt.plot(mu2[0],mu2[1],color='b',marker='.',markersize=20,markeredgecolor='b')
   plt.plot(mu3[0],mu3[1],color='m',marker='.',markersize=20,markeredgecolor='m')
   plt.plot(mu[0],mu[1],color='k',marker='.',markersize=20,markeredgecolor='k')
   plt.title('3-classes 2-dimensional data(2nd dimension vs. 1st dimension)')
   plt.xlabel('x_1')
   plt.ylabel('x_2')
   plt.grid(True)
   
   # Plotting projected data
   mirror=-1
   plt.subplot(3,1,2)
   plt.scatter(mirror*np.asarray(Yc1[0,:]),np.asarray(Yc1[1,:]),color='r',marker='o')
   plt.scatter(mirror*np.asarray(Yc2[0,:]),np.asarray(Yc2[1,:]),color='b',marker='^')
   plt.scatter(mirror*np.asarray(Yc3[0,:]),np.asarray(Yc3[1,:]),color='m',marker='+')
   plt.plot(mirror*Yc1_mu[0],Yc1_mu[1],color='r',marker='.',markersize=20,markeredgecolor='r')
   plt.plot(mirror*Yc2_mu[0],Yc2_mu[1],color='b',marker='.',markersize=20,markeredgecolor='b')
   plt.plot(mirror*Yc3_mu[0],Yc3_mu[1],color='m',marker='.',markersize=20,markeredgecolor='m')
   plt.plot(mirror*Yc_mu[0],Yc_mu[1],color='k',marker='.',markersize=20,markeredgecolor='k')
   plt.title('Projected data onto the 2 largest LDA vectors')
   plt.xlabel('Y1')
   plt.ylabel('Y2')
   
   # Projection onto the first LDA vector only
   plt.subplot(3,1,3)
   Yc1_w1 = W[:,0].T*xc1
   Yc2_w1 = W[:,0].T*xc2
   Yc3_w1 = W[:,0].T*xc3
   plt.hist(mirror*Yc1_w1,10,facecolor='r')
   plt.hist(mirror*Yc2_w1,10,facecolor='b')
   plt.hist(mirror*Yc3_w1,10,facecolor='m')
   plt.plot(mirror*Yc1_w1,np.zeros((1,n1)),color='r',marker='o')
   plt.plot(mirror*Yc2_w1,np.zeros((1,n2)),color='b',marker='^')
   plt.plot(mirror*Yc3_w1,np.zeros((1,n3)),color='m',marker='*')
   plt.title('LDA projected classes onto the biggest eigenvector ONLY')
   plt.xlabel('Y')
   plt.ylabel('Ocurrences')
   
   plt.subplots_adjust(hspace=1)
   
   # Solve classification problem
   if classify==True and mode == 'iris':
      
      new_samples = [np.matrix('5.2;4.0;1.5;0.3'),
                     np.matrix('6.6;2.2;4.1;1.9'),
                     np.matrix('6.4;3.1;4.5;1.1')]
      new_samples_proj = [W[:,0].T*sample for sample in new_samples]
      flower_classes = {0:'Setosica',1:'Versicolor',2:'Virginica'}
      for index_sample,value in enumerate(new_samples_proj):
         value = float(value)
         min_mu = 100000
         min_index = 10
         for index,mu in enumerate([Yc1_mu,Yc2_mu,Yc3_mu]):
            distance = np.linalg.norm(value-mu)
            if distance < min_mu:
               min_mu = distance
               min_index = index
         print('New sample {} is from class {}'.format(index_sample+1,flower_classes[min_index]))
            
      print('Classification finised')
   