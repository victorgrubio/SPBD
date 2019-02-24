# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.mlab import PCA as mlabPCA
import csv

class Arrow3D(FancyArrowPatch):
   def __init__(self, xs, ys, zs, *args, **kwargs):
      FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
      self._verts3d = xs, ys, zs
   def draw(self, renderer):
      xs3d, ys3d, zs3d = self._verts3d
      xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
      self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
      FancyArrowPatch.draw(self, renderer)

def analysisPCA(all_samples,m,n,k):
   mean_vector = []
   for axis in all_samples:
      mean_axis = np.mean(axis)
      mean_vector += [mean_axis]
   mean_vector = np.array(mean_vector)
   
   print('Vector de Medias:\n', mean_vector)
   
   scatter_matrix = np.zeros((m,m))
   for i in range(all_samples.shape[1]):
      scatter_matrix += (all_samples[:,i].reshape(m,1) - mean_vector).dot((all_samples[:,i].reshape(m,1) - mean_vector).T)
   
   print('Matriz de dispersión:\n', scatter_matrix)
   
   cov_mat = scatter_matrix/(all_samples.shape[1]-1)
   print('Matriz de Covarianza:\n', cov_mat)
   eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
   eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
   
   for i in range(len(eig_val_sc)):
      eigvec_sc = eig_vec_sc[:,i].reshape(1,m).T
      eigvec_cov = eig_vec_cov[:,i].reshape(1,m).T
      assert eigvec_sc.all() == eigvec_cov.all(), 'Los Autovectores no son idénticos'
      print('Autovector {}: \n{}'.format(i+1, eigvec_sc))
      print('Autovalor {} de la matriz de dispersión Matrix {}'.format(i+1, eig_val_sc[i]))
      print('Autovalor {} de la matriz de covarianza: {}'.format(i+1, eig_val_cov[i]))
      print('Factor de Escala: ', eig_val_sc[i]/eig_val_cov[i] )
      print(40 * '-')
      
   fig = plt.figure(2, figsize=(7,7))
   ax = fig.add_subplot(111, projection='3d')
   ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o',
           markersize=8, color='green', alpha=0.2)
   ax.plot([mean_vector[0]], [mean_vector[1]], [mean_vector[2]], 'o', markersize=10, color='red', alpha=0.5)
   
   if k == 3:
      for v in eig_vec_sc.T:
         a = Arrow3D([mean_vector[0], v[0]+mean_vector[0]],
                     [mean_vector[1], v[1]+mean_vector[1]],
                     [mean_vector[2], v[2]+mean_vector[2]],
                     mutation_scale=20, lw=3,arrowstyle=" -|> ", color="r")
         ax.add_artist(a)
         
         ax.set_xlabel('valores_x')
         ax.set_ylabel('valores_y')
         ax.set_zlabel('valores_Z')
         plt.title('Autovectores')
         plt.show()
   
   eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
   eig_pairs.sort()
   eig_pairs.reverse()
   
   # visual confirmation.
   for i in eig_pairs:
      print(i[0])
   
   matrix_w=np.zeros((m,k))
   for i in range(k):
      matrix_w[:,i]=eig_pairs[i][1]
   print ('Matriz W:\n', matrix_w)
   
   transformed = matrix_w.T.dot(all_samples)
   assert transformed.shape == (k,n), "La matriz no es 2x40 dimensional."
   plt.figure(3, figsize=(8,5))
   plt.plot(transformed[0,0:(n//2)], transformed[1,0:(n//2)], 'o',
                        markersize=7, color='blue', alpha=0.5, label='clase1')
   plt.plot(transformed[0,(n//2):n], transformed[1,(n//2):n], '^',
                        markersize=7, color='red', alpha=0.5, label='clase2')
   plt.xlim([-5,5])
   plt.ylim([-5,5])
   plt.xlabel('valores_x')
   plt.ylabel('valores_y')
   plt.legend()
   plt.title('Muestras transformadas con etiquetas de clase')
   plt.show()
   
   # Para evitar reescalados, se ha de llamar a matplotlib.mlab.PCA -> (standardize=False)
   mlab_pca = mlabPCA(all_samples.T,standardize=False)
   print ('Ejes PC en términos de los ejes de medida escalados por la desviaciones típicas\n',mlab_pca.Wt)
   
   
   plt.figure(4,figsize=(8,16))
   plt.subplot(k,1,1)
   # plt.hold(True)
   plt.plot(mlab_pca.Y[0:(n//2),0],mlab_pca.Y[0:(n//2),1],'o', 
            markersize=7, color='blue', alpha=0.5, label='clase1')
   plt.plot(mlab_pca.Y[(n//2):n,0], mlab_pca.Y[(n//2):n,1],'^',
            markersize=7, color='red', alpha=0.5, label='clase2')
   
   plt.ylabel('valores_y')
   plt.xlim([-4,4])
   plt.ylim([-4,4])
   plt.legend()
   
   plt.title('Muestras transformadas mediante matplotlib.mlab.PCA()')
   #plt.hold(False)
   # Depending on the way, eigenvectors can be positives or negatives
   transformed[1] = transformed[1]*(-1) # invertimos el eje PC2
   plt.subplot(2,1,2)
   #plt.hold(True)
   plt.plot(transformed[0,0:(n//2)], transformed[1,0:(n//2)], 'o',
                        markersize=7, color='blue', alpha=0.5, label='clase1')
   plt.plot(transformed[0,(n//2):n], transformed[1,(n//2):n], '^',
                        markersize=7, color='red', alpha=0.5, label='clase2')
   plt.xlim([-5,4])
   plt.ylim([-4,4])
   plt.xlabel('valores_x')
   plt.ylabel('valores_y')
   plt.legend()
   plt.title('Muestras transformadas mediante el procedimiento paso-a-paso')
   #plt.hold(False)
   plt.show()
   #plt.hold()

"""
Solving for Iris Dataset
"""

"""
Main method
"""

if __name__ == '__main__':
   
   #First we define the values of N,M and K
   n = 40
   m = 3
   k = m-1 #Value of k based on the features (M)
         
   np.random.seed(1) # random seed for consistency
   mu_vec1  = np.array([0]*m)
   cov_mat1 = np.identity(m)
   
   class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, n//2).T
   assert class1_sample.shape == (m,n//2), "La matriz no tiene dimensiones 3x20"
   
   mu_vec2  = np.array([1]*m)
   cov_mat2 = np.identity(m)
   class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, n//2).T
   assert class1_sample.shape == (m,n//2), "La matriz no tiene dimensiones 3x20"
   
   fig = plt.figure(1, figsize=(8,8))
   ax = fig.add_subplot(111, projection='3d')
   plt.rcParams['legend.fontsize'] = 10
   ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:],
   'o', markersize=8, color='blue', alpha=0.5, label='class1')
   ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
   '^', markersize=8, alpha=0.5, color='red', label='class2')
   plt.title('Samples for class 1 and class 2')
   ax.legend(loc='upper right')
   plt.show()
   
   all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
   assert all_samples.shape == (m,n), "La matriz no tiene dimensiones 3x40"
   
   #After the previous data generation, we begin with the analysis
   analysisPCA(all_samples,m,n,k)
   
   #Reading data from iris dataset
   iris_samples=[]
   archivo='iris_data.csv'
   f=open(archivo,'r')
   texto = csv.reader(f,delimiter=',')
   for line in texto:
       for i in range(0,4):
            line[i]=float(line[i])
       iris_samples.append(line[0:4])
   f.close()  
   iris_samples=np.asarray(iris_samples).T   
   
   m_iris = iris_samples.shape[0]
   n_iris = iris_samples.shape[1]
   k_iris = m_iris - 1
   print('\n'.join(['---------------------------',
                    'PCA APPLIED TO IRIS DATASET',
                    '----------------------------']))
   #Solving for Iris dataset
   analysisPCA(iris_samples,m_iris,n_iris,k_iris)