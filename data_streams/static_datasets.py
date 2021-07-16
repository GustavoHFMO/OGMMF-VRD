'''
Created on 4 de jun de 2018
@author: gusta
'''
import numpy as np
from numpy import array
from sklearn import datasets

class Datasets:
    def __init__(self):
        pass
        
    def chooseDataset(self, cod):
        '''
        method to choose one static dataset
        :param: cod: number of the respective dataset
        :return: data with patterns and labels
        '''
        
        if(cod == 1):
            X = np.asarray([[-11, -20.5], [-10.5, -15.3], [-9.7, -12.1], 
                                [-1, -0.5], [-0.5, -0.3], [-0.7, -0.1], 
                                [0.5, 2], [0.5, 1.7], [0.5, 1.5], 
                                [2.6, 2.2], [2.4, 1.7], [2.4, 1.8], 
                                [3.1, -1.1], [3.3, -0.7], [2.8, -0.5]])
            y = np.asarray([1, 1, 1, 
                                  2, 2, 2,
                                  3, 3, 3,
                                  4, 4, 4,
                                  5, 5, 5])
            return X, y
        
        elif(cod == 2):
            iris = datasets.load_iris()
            X = iris.data[:, :2]
            X = 3 * X
            y = iris.target
            return X, y
        
        elif(cod == 3):
            from sklearn.datasets.samples_generator import make_blobs
            X, y = make_blobs(n_samples=200, centers=4,
                           cluster_std=0.60, random_state=0)
            X = X[:, ::-1] # flip axes for better plotting
            return X, y
        
        elif(cod == 4):
            X, y = datasets.make_moons(n_samples=400, noise=.05)
            X = X[:, ::-1] # flip axes for better plotting
            return X, y
        
        elif(cod == 5):
            wine = datasets.load_wine()
            X = wine.data[:, :2]
            X = 3 * X
            y = wine.target
            return X, y
        
        elif(cod == 6):
            cancer = datasets.load_breast_cancer()
            X = cancer.data[:, :2]
            X = 3 * X
            y = cancer.target
            return X, y
        
        elif(cod == 7):
            from sklearn.datasets.samples_generator import make_gaussian_quantiles
            X, y = make_gaussian_quantiles(n_samples=400, n_features=2, n_classes=3)
            return X, y
        
        
