'''
Created on 22 de ago de 2018
'''

from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from detectors.ewma import EWMA
from detectors.ddm import DDM
from detectors.eddm import EDDM
from detectors.adwin import ADWINChangeDetector
al = Adjust_labels()
import numpy as np
import time
import copy

class Hoeffding_Detector(PREQUENTIAL_SUPER):
    def __init__(self, window_size=200, drift_detector="ADWIN"):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        # main variables
        self.BASE_LEARNER = HoeffdingAdaptiveTreeClassifier(leaf_prediction='mc', random_state=1)
        self.DETECTOR = self.selectDriftDetector(drift_detector, window_size)
        self.WINDOW_SIZE = window_size
        self.LOSS_STREAM = []
        self.DETECTIONS = [0]
        self.WARNINGS = [0]
        self.CLASSIFIER_READY = True
        
        # auxiliar variables
        self.NAME = 'Hoeffding_tree_' + drift_detector
        self.TARGET = []
        self.PREDICTIONS = []
        self.count = 0
    
    '''
    WINDOW MANAGEMENT
    '''
    
    def selectDriftDetector(self, typ, window_size):
        '''
        Method to select the drift detector to be used
        :typ: the name of the drift detector
        '''
        if(typ == "EWMA"):
            return EWMA(min_instance=window_size, c=1, w=0.5)
        elif(typ == "DDM"):
            return DDM(min_instance=window_size)
        elif(typ == "EDDM"):
            return EDDM(min_instance=window_size)
        elif(typ == "ADWIN"):
            return ADWINChangeDetector()
    
    def transferKnowledgeWindow(self, W, W_warning):    
        '''
        method to transfer the patterns of one windown to other
        :param: W: window that will be updated
        :param: W_warning: window with the patterns that will be passed
        '''
        
        W = W_warning
        
        return W
    
    def manageWindowWarning(self, W, x):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        if(self.CLASSIFIER_READY):
            W = self.incrementWindow(W, x)
            
            if(len(W) > self.WINDOW_SIZE/2):
                W = self.resetWindow(W)
        
        return W
     
    def resetWindow(self, W):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        return np.array([])
    
    def incrementWindow(self, W, x):
        '''
        method to icrement the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * (len(W)+1)
        aux[:-1] = W
        aux[-1] = x
        
        return np.asarray(aux) 
    
    def slidingWindow(self, W, x):
        '''
        method to slide the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * len(W)
        aux[0:-1] = W[1:]
        aux[-1] = x
    
        return np.asarray(aux)

    '''
    FIT THE CLASSIFIER
    '''
    
    def trainClassifier(self, W):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''
        
        # starting the classifier
        self.CLASSIFIER = copy.deepcopy(self.BASE_LEARNER)
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # fitting the dataset
        for x, y in zip(x_train, y_train):
            self.CLASSIFIER.partial_fit([x], [y])

        # returning the new classifier        
        return self.CLASSIFIER
    
    '''
    RUN ON DATA STREAM
    '''
    
    def run(self, labels, stream, cross_validation=False, fold=5, qtd_folds=30):
        '''
        method to run the stream
        '''
        
        # starting time
        self.start_time = time.time()
        
        ######################### 1. FITTING THE STREAM AND AUXILIAR VARIABLES ####################
        # defining the stream
        self.STREAM = stream
                
        # obtaining the initial window
        W = self.STREAM[:self.WINDOW_SIZE]
        
        # instantiating a window for warning levels 
        W_warning = []
        ######################### 1. #############################################################
        
         
         
         
        ########################### 2. STARTING THE CLASSIFIER AND DETECTOR #########################
        # training the classifier
        self.CLASSIFIER = self.trainClassifier(W)
        
        # fitting the detector
        self.DETECTOR.fit(self.CLASSIFIER, W) 
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
                x, y = X[0:-1], int(X[-1])
        ##################################### 3. ######################################################
        
        
        
                ########################################## 4. ONLINE CLASSIFICATION ###################################
                # using the classifier to predict the class of current label
                yi = self.CLASSIFIER.predict([x])[0]
                
                # storing the predictions
                self.PREDICTIONS.append(yi)
                self.TARGET.append(y)
                ########################################## 4. #########################################################
                
                
                ################################ 4.1. UPDATING THE TREE ONLINE #############################################
                self.CLASSIFIER.partial_fit([x], [y])        
                ################################ 4.1. ######################################################################
                
                
                ################################ 7. MONITORING THE DRIFT  ##############################################
                # verifying the classifier
                if(self.CLASSIFIER_READY):
    
                    # sliding the current observation into W
                    W = self.slidingWindow(W, X)
                
                    # monitoring the datastream
                    warning_level, change_level = self.DETECTOR.detect(y, yi)
                ################################## 7. ####################################################################
                
                
                    ################################## 8. WARNING ERROR PROCEDURES ###########################################
                    if(warning_level):
    
                        # managing the window warning
                        W_warning = self.manageWindowWarning(W_warning, X)
                        
                        # storing the time when warning was triggered
                        self.WARNINGS.append(i)
                    ################################## 8. ####################################################################
                    
                        
                
                    ################################## 9. DRIFT ERROR PROCEDURES ############################################
                    if(change_level):
                        
                        # storing the time of change
                        self.DETECTIONS.append(i)
                        
                        # reseting the detector
                        self.DETECTOR.reset()
                            
                        # reseting the window
                        W = self.transferKnowledgeWindow(W, W_warning)
                        
                        # reseting the classifier 
                        self.CLASSIFIER_READY = False
                    ################################## 9. ####################################################################
                    
                    
                
                ################################## 10. COLLECTING NEW DATA ############################################
                elif(self.WINDOW_SIZE > len(W)):
                    
                    # sliding the current observation into W
                    W = self.incrementWindow(W, X)
                ################################## 10. ################################################################
                
                
                
                ################################## 11. RETRAINING THE CLASSIFIER #########################################
                else:
                    
                    # to remodel the knowledge of the classifier
                    self.CLASSIFIER = self.trainClassifier(W)
                        
                    # fitting the detector
                    self.DETECTOR.fit(self.CLASSIFIER, W) 
                            
                    # releasing the new classifier
                    self.CLASSIFIER_READY = True
                ################################## 11. ###################################################################
                
                    
                # print the current process
                self.printIterative(i)
                
        # ending time
        self.end_time = time.time()
    
def main():
    
    
    ####### 1. DEFINING THE DATASETS ##################################################################
    #i = 2
    # REAL DATASETS 
    #dataset = ['PAKDD', 'elec', 'noaa', 'poker-lsn2', 'covtypeNorm2']
    #labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    
    # SYNTHETIC DATASETS 
    i = 0
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    ####### 1. ########################################################################################
    stream_records = al.adjustStream(labels, stream_records)
    
    ####### 2. DEFINING THE MODEL PARAMETERS ##########################################################
    # instantiate the prequetial
    preq = Hoeffding_Detector(window_size=50, drift_detector="ADWIN")
    preq.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
    
    # presenting the accuracy
    print("Accuracy: ", preq.accuracyGeneral())
    preq.plotAccuracy()
    ####### 2. DEFINING THE MODEL PARAMETERS ##########################################################
    
    
    ####### 3. STORING THE PREDICTIONS ################################################################
    import pandas as pd
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS, 'target':preq.TARGET})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+".csv")
    ####### 3. STORING THE PREDICTIONS ################################################################
    
if __name__ == "__main__":
    main()        
           
        
    