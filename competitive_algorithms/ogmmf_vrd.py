'''
Created on 16 de jul de 2021
By Gustavo Oliveira
Universidade Federal de Pernambuco, Recife, Brasil
E-mail: ghfmo@cin.ufpe.br
ALGORITHMS USED IN THE PAPER PUBLISHED BELOW:
Oliveira, Gustavo, Leandro Minku, and Adriano Oliveira. 
"Tackling Virtual and Real Concept Drifts: An Adaptive Gaussian Mixture Model.
" arXiv preprint arXiv:2102.05983 (2021).
url:https://arxiv.org/abs/2102.05983
'''

from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
from gaussian_models.gmm_unsupervised import Gaussian
from gaussian_models.gmm_unsupervised import GMM_KDN
from imblearn.metrics import geometric_mean_score 
from sklearn.metrics import accuracy_score
from detectors.eddm import EDDM
al = Adjust_labels()
import numpy as np
import copy
import time

class GMM_VD(GMM_KDN):
    def __init__(self):
        super().__init__()
        self.noise_threshold = 0.8
        self.n_vizinhos = 5

    '''
    METHOD INITIALIZATION
    '''
    
    def start(self, train_input, train_target, noise_threshold=False, n_vizinhos=False):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # training the GMM
        self.fit(train_input, train_target, noise_threshold, n_vizinhos)
         
        ######## activating online mechanisms ########           
        # defining the theta
        self.computeTheta()
        
        # defining the sigma
        self.computeSigma()
            
    def computeTheta(self):
        '''
        Method to define the theta value
        :x: the input data
        :y: the respective target
        '''
        
        # computing the pertinence of each observation
        pertinencia = []
        for i in self.train_input:
            pertinencia.append(self.predictionProb(i))
        
        # computing the minimum for each class
        self.minimum_classes = []
        indices = []
        
        # for to iterate by each class
        for i in self.unique:
            
            # variables to store the pertinence by class and the indexes of each observation
            pertinences_by_class = []
            indexes = []
            
            # for to iterate by each observation in the training set
            for x, j in enumerate(self.train_target):
                if(i == j):
                    
                    # storing each pertinence
                    pertinences_by_class.append(pertinencia[x])
                    # storing the indexes for each observation
                    indexes.append(x)
                    
            # getting the smaller pertinence by class
            self.minimum_classes.append(np.min(pertinences_by_class))
            # getting the indexes for the furthest observations
            indices.append(indexes[np.argmin(pertinences_by_class)])
            
        # defining the theta value
        self.theta = np.min(self.minimum_classes) 
      
    def computeSigma(self):
        '''
        Method to define the sigma value
        '''

        # computing the max and minimum value for the training set
        x_max = np.max(self.train_input)
        x_min = np.min(self.train_input)
         
        # computing the sigma value
        self.sigma = (x_max - x_min)/20
        
    '''
    VIRTUAL ADAPTATION
    '''
    
    def virtualAdaptation(self, x, y, W, t):
        '''
        method to update an gaussian based on error
        :x: current pattern
        :y: true label of pattern
        :W: validation dataset
        :t: time
        '''
        
        if(self.noiseFiltration(x, y, W)):
            
            # find the nearest gaussian
            prob, gaussian = self.nearestGaussian(x, y)
    
            # to update the nearest gaussian            
            self.updateGaussian(x, gaussian)
                    
            # to create a gaussian                    
            if(self.theta > prob):
                self.createGaussian(x, y)
                self.updateTheta(y, prob)
                print("number of gaussians:", len(self.gaussians))
                
    def noiseFiltration(self, x, y, W, plot=False):
        '''
        Method to filter noisy observations
        :x: current pattern
        :y: true label of pattern
        :W: validation dataset
        '''
        
        # adjusting the window
        W = np.asarray(W)
                                
        # updating the data for train and target
        self.train_input = W[:,0:-1]
        self.train_target = W[:,-1]
            
        # to verify if the instance is an noisy
        if(self.kDNIndividual(x, y, self.train_input, self.train_target, plot) < self.noise_threshold):
            return True
        else:
            return False
            
    '''
    ONLINE LEARNING
    '''
        
    def nearestGaussian(self, x, y):
        '''
        method to find the nearest gaussian of the observation x
        :x: observation 
        :y: label
        '''
        
        # receiving the gaussian with more probability for the pattern
        z = [0] * len(self.gaussians)
        for i in range(len(self.gaussians)):
            if(self.gaussians[i].label == y):
                z[i] = self.conditionalProbability(x, i)

        # nearest gaussian
        gaussian = np.argmax(z)
        
        # returning the probability and the nearest gaussian
        return z[gaussian], gaussian
        
    def updateGaussian(self, x, gaussian):
        '''
        method to update the nearest gaussian of x
        :x: the observation that will be used to update a gaussian
        :gaussian: the number of gaussian that will be updated  
        '''

        # updating the likelihood of all gaussians for x
        self.updateLikelihood(x)
        
        # updating the gaussian weights
        self.updateWeight()
                
        # storing the old mean
        old_mean = self.gaussians[gaussian].mu
        
        # updating the mean
        self.gaussians[gaussian].mu = self.updateMean(x, gaussian)

        # updating the covariance        
        self.gaussians[gaussian].sigma = self.updateCovariance(x, gaussian, old_mean)
        
    def updateLikelihood(self, x):
        '''
        method to update the parameter cver
        :param: x: new observation
        '''
        
        # getting the probabilities
        probabilities = self.posteriorProbabilities(x)
        
        # updating the loglikelihood
        for i in range(len(self.gaussians)):
            self.gaussians[i].dens += probabilities[i]
        
    def updateMean(self, x, gaussian):
        '''
        Method to update the mean of a gaussian i
        :x: pattern
        :gaussian: gaussian number
        return new mean
        '''
        
        # computing the new mean
        part1 = self.posteriorProbability(x, gaussian)/self.gaussians[gaussian].dens
        part2 = np.subtract(x, self.gaussians[gaussian].mu)
        new = self.gaussians[gaussian].mu + (np.dot(part1, part2))
        
        # returning mean
        return new
    
    def updateCovariance(self, x, i, old_mean):
        '''
        Method to update the mean of a gaussian i
        :x: pattern
        :gaussian: gaussian number
        return new mean
        '''
        
        # equation to compute the covariance
        ######## primeira parte ##############
        # sigma passado
        part0 = self.gaussians[i].sigma
        
        # primeiro termo
        part1 = np.subtract(self.gaussians[i].mu, old_mean)
        
        # segundo termo transposto
        part2 = np.transpose([part1])
        
        # multiplicacao dos termos
        part3 = np.dot(part2, [part1])
        
        # subtracao do termo pelo antigo
        part4 = np.subtract(part0, part3)
        ########################################
        
        
        ######## segunda parte ##############
        #ajuste de pertinencia
        part5 = self.posteriorProbability(x, i)/self.gaussians[i].dens
        
        # primeiro termo
        part6 = np.subtract(x, self.gaussians[i].mu)
        
        # segundo termo transposto
        part7 = np.transpose([part6])
        
        # multiplicacao do primeiro pelo segundo
        part8 = np.dot(part7, [part6])
        
        # subtracao do sigma antigo pelos termos
        part9 = np.subtract(part8, part0)
        
        
        # multiplicacao da pertinencia pelo colchetes
        part10 = np.dot(part5, part9)
        ########################################
        
        
        #final
        covariance = np.add(part4, part10) 
        
        # returning covariance
        return covariance

    '''
    CREATING GAUSSIANS ONLINE
    '''
    
    def createGaussian(self, x, y):
        '''
        method to create a new gaussian
        :x: observation 
        :y: label
        '''
        
        # mu
        mu = x
        
        # covariance
        cov = (self.sigma**2) * np.identity(len(x))
        
        # label
        label = y
        
        # new gaussian
        g = Gaussian(mu, cov, 1, label)
        
        # adding the new gaussian in the system
        self.gaussians.append(g)
        
        # adding 
        self.K += 1
        
        # updating the density of all gaussians
        self.updateLikelihood(x)
        
        # updating the weights of all gaussians
        self.updateWeight()
        
    def updateTheta(self, y, prob):
        '''
        method to update the theta value
        :x: current observation
        :y: respective class
        '''

        # for to update the classes already seen
        if(y not in self.unique):
            # adding the new prob
            self.unique.append(y)
            self.minimum_classes.append(prob)
        else:
            # updating the prob for the current class
            index = self.unique.index(y)
            self.minimum_classes[index] = prob
        
        # updating the theta value
        self.theta = np.min(self.minimum_classes)
    
class OGMMF_VRD(PREQUENTIAL_SUPER):
    def __init__(self, window_size=200, noise_threshold=0.8, metric="accuracy"):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        # main variables
        self.CLASSIFIER = GMM_VD()
        self.CLASSIFIER.noise_threshold = noise_threshold
        self.DETECTOR = EDDM(min_instance=window_size, C=1, W=0.5)
        self.METRIC = metric
        self.WINDOW_SIZE = window_size
        self.LOSS_STREAM = []
        self.DETECTIONS = [0]
        self.WARNINGS = [0]
        self.MEMORY = []
        self.POOL_SIZE = 20
        self.RETRAIN_SIZE = int(self.WINDOW_SIZE * 0.25)
        self.CLASSIFIER_READY = True
        self.WARNING_SIGNAL = False
        
        self.NAME = 'OGMMF-VRD'
        
        # auxiliar variable
        self.PREDICTIONS = []
        self.TARGET = []
        self.count = 0
    
    '''
    METHODS TO TRAIN THE CLASSIFIERS
    '''
        
    def trainClassifier(self, W):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # fitting the dataset
        self.CLASSIFIER.start(x_train, y_train)

        # printing the accuracy for training set
        pred = self.CLASSIFIER.predict(x_train)
        print(accuracy_score(y_train, pred))
        
        # returning the new classifier        
        return self.CLASSIFIER
    
    def trainNewClassifier(self, W):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # new classifier
        CLASSIFIER = GMM_VD()
        
        # fitting the dataset
        CLASSIFIER.start(x_train, y_train)

        # returning the new classifier        
        return CLASSIFIER
    
    
    '''
    METHODS TO MANAGE THE DATA
    '''
    
    def transferKnowledgeWindow(self, W, W_warning):    
        '''
        method to transfer the patterns of one windown to other
        :param: W: window that will be updated
        :param: W_warning: window with the patterns that will be passed
        '''
        
        W = W_warning
        
        return W
    
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
    
    '''
    METHODS STORE AND EVALUATE MODELS
    '''
    
    def storeClassifier(self, classifier):
        '''
        Method to store a classifier into a pool
        '''

        # storing the classifier
        self.MEMORY.append(copy.deepcopy(classifier))
        
        # deleting the old classifier
        if(len(self.MEMORY) > self.POOL_SIZE):
            del self.MEMORY[0]
        
    def estimateClassifier(self, W):
        '''
        method to estimate the best classifier to the current moment
        '''
        
        # split the input and target
        X, Y = W[:,0:-1], W[:,-1]
        
        # estimating the best classifier
        errors = []
        for classifier in self.MEMORY:
            YI = classifier.predict(X)
            if(self.METRIC=="accuracy"):
                errors.append(accuracy_score(Y, YI))
            elif(self.METRIC=="gmean"):
                errors.append(geometric_mean_score(Y, YI))
            
        # returning the best classifier
        return copy.deepcopy(self.MEMORY[np.argmax(errors)])
        
    '''
    METHODS RUN THE ALGORITHM
    '''
    
    def run(self, labels, stream, cross_validation=False, fold=5, qtd_folds=30):
        '''
        method to run the stream
        '''
        
        # starting time
        self.start_time = time.time()
        
        ######################### 1. FITTING THE STREAM AND AUXILIAR VARIABLES ####################
        # storing the new stream
        self.STREAM = stream
        
        # obtaining the initial window
        W = self.STREAM[:self.WINDOW_SIZE]
        
        # instantiating the validation window
        W_validation = W
        ######################### 1. ############################################################# 
        
        
        
        
        ########################### 2. STARTING THE CLASSIFIER AND DETECTOR #########################
        # training the classifier
        self.CLASSIFIER = self.trainClassifier(W)
        
        # storing the new classifier
        self.storeClassifier(self.CLASSIFIER)
            
        # fitting the detector
        self.DETECTOR.fit(self.CLASSIFIER, W)
        
        # instantiating a window for warning levels
        W = [] 
        W_warning = [] 
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
                yi = self.CLASSIFIER.predict(x)
                
                # storing the predictions
                self.PREDICTIONS.append(yi)
                self.TARGET.append(y)
                ########################################## 4. #########################################################


                
                ########################################## 5. VIRTUAL ADAPTATION #######################################
                # sliding the current observation into W
                W_validation = self.slidingWindow(W_validation, X)
                    
                # updating the gaussian if the classifier miss
                self.CLASSIFIER.virtualAdaptation(x, y, W_validation, i)
                ######################################### 5. ###########################################################
                
                
                
                
                ################################ 7. MONITORING THE DRIFT  ##############################################
                # verifying the claassifier
                if(self.CLASSIFIER_READY):
    
                    # monitoring the datastream
                    warning_level, change_level = self.DETECTOR.detect(y, yi)
                ################################## 7. ####################################################################
                
                
                
                    ################################## 8. WARNING ERROR PROCEDURES ###########################################
                    # trigger the warning VIRTUAL
                    if(warning_level):
                        
                        # storing the time when warning was triggered
                        self.WARNINGS.append(i)
                        
                        # activate window collection
                        self.WARNING_SIGNAL = True
                    
                        
                    elif(self.WARNING_SIGNAL):

                        # managing the window warning
                        W_warning = self.manageWindowWarning(W_warning, X)
                    ################################## 8. ####################################################################
                        
                        
                        
                        
                    ################################## 9. DRIFT ERROR PROCEDURES ############################################
                    # trigger the change VIRTUAL    
                    if(change_level):
                        # storing the time of change
                        self.DETECTIONS.append(i)
                        
                        # reseting the detector
                        self.DETECTOR.reset()
                            
                        # reseting the window
                        W = self.transferKnowledgeWindow(W, W_warning)
                        
                        # reseting the classifier 
                        self.CLASSIFIER_READY = False
                        
                        # deactivate warning signal
                        self.WARNING_SIGNAL = False
                    ################################## 9. ####################################################################
                    
                    
                    
                ################################## 10. COLLECTING NEW DATA ############################################    
                elif(self.WINDOW_SIZE > len(W)):
                    
                    # sliding the current observation into W
                    W = self.incrementWindow(W, X)
                ################################## 10. ################################################################
                
                
                
                    ################################## 11. REUSING THE CLASSIFIER FROM POOL ##################################
                    # to know if the window has enough data
                    if(len(W) == self.RETRAIN_SIZE):
                            
                        # training a new classifier
                        self.storeClassifier(self.trainNewClassifier(W))

                        # estimating a new classifier
                        self.CLASSIFIER = self.estimateClassifier(W)
                    ################################## 11. ###################################################################
                    
                    
                
                
                ################################## 11. RETRAINING THE CLASSIFIER #########################################
                else:
                    # to remodel the knowledge of the classifier
                    self.CLASSIFIER = self.trainClassifier(W)
                    
                    # storing the new classifier
                    self.storeClassifier(self.CLASSIFIER)
                    
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
    
    i = 0
    
    '''
    ############################################ REAL DATASETS ##################################################
    dataset = ['gassensor', 
               'INSECTS-gradual_balanced_norm', 
               'INSECTS-incremental-abrupt_balanced_norm',
               'INSECTS-incremental-reoccurring_balanced_norm',
               'INSECTS-abrupt_balanced_norm',
               'PAKDD', 
               'elec', 
               'noaa']
    labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    stream_records = al.adjustStream(labels, stream_records)
    #############################################################################################################
    '''
    
    ############################################ SYNTHETIC DATASETS #############################################
    dataset = ['circles', 
               'sine1', 
               'sine2', 
               'virtual_5changes', 
               'virtual_9changes', 
               'SEA', 
               'SEARec']
    
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    stream_records = al.adjustStream(labels, stream_records)
    #############################################################################################################

    #4. instantiate the prequetial
    preq = OGMMF_VRD()
    
    #5. execute the prequential
    preq.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
    preq.plotAccuracy()
    
    # printing the final accuracy
    print("Accuracy: ", preq.accuracyGeneral())

    # storing only the predictions
    import pandas as pd
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS, 'target': preq.TARGET})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+".csv")
    
if __name__ == "__main__":
    main()        
           
    