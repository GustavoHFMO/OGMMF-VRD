'''
Created on 11 de out de 2018
@author: gusta
'''

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class PREQUENTIAL_SUPER():
    def __init__(self):
        '''
        Class for control the comparative algorithms
        '''
        
        self.NAME = ''
        self.TARGET = []
        self.PREDICTIONS = []
    
    def returnTarget(self):
        '''
        method to return only the target o
        '''
        
        return self.TARGET
    
    def returnPredictions(self):
        '''
        method to return only the predictions
        '''
        
        return np.asarray(self.PREDICTIONS).astype('float64')
    
    def accuracyGeneral(self):
        '''
        method to return the system accuracy for the stream
        '''
        
        y_true = self.returnTarget()
        y_pred = self.returnPredictions()
                            
        return accuracy_score(y_true, y_pred)

    def timeExecution(self):
        '''
        method to return the system accuracy for the stream
        '''
        
        start = self.start_time
        end = self.end_time
                            
        return end-start

    def printIterative(self, i):
        '''
        method to show iteratively the current accuracy 
        '''
        current_accuracy = accuracy_score(self.TARGET, self.PREDICTIONS)*100
        percent_instances = (i*100)/len(self.STREAM)
        string = self.NAME+": %.2f -> (%d) %.2f of instances processed" % (current_accuracy, i, percent_instances)
        
        print(string)
    
    def plotAccuracy(self):
        '''
        Method to plot the current accuracy
        '''

        # importing the class to plot the accuracy
        import ilustrations.generate_accuracy as ga
        
        # calculating the timeseries
        timeSeries = ga.calculateLongAccuracy(self.returnTarget(), self.returnPredictions(), 300)
        
        # plotting the accuracy
        plt.plot(timeSeries)
        plt.show()
    
    def cross_validation(self, i, qtd_folds, fold):
        '''
        Method to use the cross validation to data streams
        '''
        
        # if the current point reach the maximum, then is reseted 
        if(self.count == qtd_folds):
            self.count = 0
            
        # false if the fold is equal to count
        if(self.count == fold):
            Flag = True
        else:
            Flag = False
        
        # each iteration is accumuled an point
        self.count += 1
        
        #returning the flag
        return Flag
        