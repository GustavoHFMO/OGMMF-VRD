'''
Created on 22 de ago de 2018
'''

from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OzaBaggingADWINClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.bayes import NaiveBayes
al = Adjust_labels()
import numpy as np
import time

class Ensemble_Learners(PREQUENTIAL_SUPER):
    def __init__(self, window_size=200, ensemble="AWE"):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        # main variables
        self.CLASSIFIER = self.selecteEnsemble(ensemble, window_size)
        self.WINDOW_SIZE = window_size
        self.LOSS_STREAM = []
        self.DETECTIONS = [0]
        self.WARNINGS = [0]
        self.CLASSIFIER_READY = True
        
        # auxiliar variables
        self.NAME = ensemble
        self.TARGET = []
        self.PREDICTIONS = []
        self.count = 0
    
    '''
    FIT THE CLASSIFIER
    '''
    
    def selecteEnsemble(self, typ, window_size):
        '''
        Method to select the drift detector to be used
        :typ: the name of the drift detector
        '''
        if(typ == "AWE"):
            return AccuracyWeightedEnsembleClassifier(n_estimators=5, 
                                                      n_kept_estimators=3,
                                                      base_estimator=NaiveBayes(),
                                                      window_size=window_size, n_splits=5)
        elif(typ == "LevBag"):
            return LeveragingBaggingClassifier(base_estimator=NaiveBayes(),
                                               n_estimators=3,
                                               random_state=112)
        
        elif(typ == "OzaAD"):
            return OzaBaggingADWINClassifier(base_estimator=NaiveBayes(), 
                                             n_estimators=3, 
                                             random_state=112)
            
        elif(typ == "OzaAS"):
            return OzaBaggingClassifier(base_estimator=NaiveBayes(), 
                                        n_estimators=3, 
                                        random_state=112)
            
        elif(typ == "ARF"):
            return AdaptiveRandomForestClassifier(n_estimators=3,
                                                  leaf_prediction='mc',
                                                  random_state=112)
    
    def trainClassifier(self, W, labels):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # fitting the dataset
        for x, y in zip(x_train, y_train):
            x = np.asarray([x])
            y = np.asarray([y])
            
            if(self.NAME == "LevBag" or self.NAME == "OzaAD" or self.NAME == "OzaAS"):
                #self.CLASSIFIER.partial_fit(x, y, classes=list(np.unique(y_train)))
                self.CLASSIFIER.partial_fit(x, y, classes=list(labels))
            else:
                self.CLASSIFIER.partial_fit(x, y)

        # returning the new classifier        
        return self.CLASSIFIER
    
    '''
    RUN ON DATA STREAM
    '''
    
    def run(self, labels, stream, cross_validation=False, fold=5, qtd_folds=30):
        '''
        method to run the stream
        '''
        
        # ....
        labels = np.unique(stream[:,-1])
        
        # starting time
        self.start_time = time.time()
        
        ######################### 1. FITTING THE STREAM AND AUXILIAR VARIABLES ####################
        # defining the stream
        self.STREAM = stream
        
        # obtaining the initial window
        W = self.STREAM[:self.WINDOW_SIZE]
        ######################### 1. #############################################################
         
         
        ########################### 2. STARTING THE CLASSIFIER AND DETECTOR #########################
        # training the classifier
        self.CLASSIFIER = self.trainClassifier(W, labels)
        ############################ 2. ##############################################################
        
        
        #################################### 3.SIMULATING THE STREAM ################################
        # for to operate into a stream
        for i, X in enumerate(self.STREAM[self.WINDOW_SIZE:]):
            
            # to use the cross validation
            run=True
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = False
            
            # to execute the prequential precedure
            if(run):
                
                # split the current example on pattern and label
                x, y = np.asarray([X[0:-1]]), np.asarray([int(X[-1])])
        ##################################### 3. ######################################################
        
        
                ########################################## 4. ONLINE CLASSIFICATION ###################################
                # using the classifier to predict the class of current label
                yi = self.CLASSIFIER.predict(x)[0]
                
                # storing the predictions
                self.PREDICTIONS.append(yi)
                self.TARGET.append(y[0])
                ########################################## 4. #########################################################
                
                
                ################################ 5. UPDATING ONLINE ######################################################
                self.CLASSIFIER.partial_fit(x, y)        
                ################################ 5. ######################################################################
                    
                # print the current process
                self.printIterative(i)
                
        # ending time
        self.end_time = time.time()
    
def main():
    
    
    ####### 1. DEFINING THE DATASETS ##################################################################
    i = 1
    dataset = ['gassensor', 'INSECTS-gradual_balanced_norm', 'INSECTS-incremental-abrupt_balanced_norm']
    labels, _, stream_records = ARFFReader.read("../data_streams/usp/"+dataset[i]+".arff")
    # REAL DATASETS 
    #dataset = ['PAKDD', 'elec', 'noaa', 'poker-lsn2', 'covtypeNorm2']
    #labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    
    # SYNTHETIC DATASETS 
    #i = 2
    #dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    #labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    stream_records = al.adjustStream(labels, stream_records)
    ####### 1. ########################################################################################
    
    
    ####### 2. DEFINING THE MODEL PARAMETERS ##########################################################
    # instantiate the prequetial
    preq = Ensemble_Learners(window_size=200, ensemble="LevBag")
    preq.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
    
    # presenting the accuracy
    preq.plotAccuracy()
    print("Accuracy: ", preq.accuracyGeneral())
    ####### 2. DEFINING THE MODEL PARAMETERS ##########################################################
    
    
    ####### 3. STORING THE PREDICTIONS ################################################################
    import pandas as pd
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS, 'target':preq.TARGET})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+".csv")
    ####### 3. STORING THE PREDICTIONS ################################################################
    
if __name__ == "__main__":
    main()        
           
        
    