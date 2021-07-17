#-*- coding: utf-8 -*-

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

# IMPORTING THE ALGORITHMS
from competitive_algorithms.ogmmf_vrd import OGMMF_VRD
from competitive_algorithms.ensemble_learners import Ensemble_Learners
from competitive_algorithms.hoeffding_detector import Hoeffding_Detector
from competitive_algorithms.gmm_vrd import GMM_VRD
from competitive_algorithms.dynse import Dynse
from competitive_algorithms.igmmcd import IGMM_CD

# Importing some libs to help the execution
from streams.readers.arff_reader import ARFFReader
from data_streams.adjust_labels import Adjust_labels
al = Adjust_labels()

####### 1. DEFINING THE DATASETS ##################################################################
i = 0
'''
# REAL DATASETS 
dataset = ['gassensor', 
            'INSECTS-gradual_balanced_norm', 
            'INSECTS-incremental-abrupt_balanced_norm',
            'INSECTS-incremental-reoccurring_balanced_norm',
            'INSECTS-abrupt_balanced_norm',
            'PAKDD', 
            'elec', 
            'noaa']
labels, _, stream_records = ARFFReader.read("data_streams/real/"+dataset[i]+".arff")
'''
 
# SYNTHETIC DATASETS 
dataset = ['circles', 
           'sine1', 
           'sine2', 
           'virtual_5changes', 
           'virtual_9changes', 
           'SEA', 
           'SEARec']
labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/"+dataset[i]+".arff")
stream_records = al.adjustStream(labels, stream_records)
####### 1. ########################################################################################
    
    
####### 2. DEFINING THE MODELS ####################################################################
# OGMMF-VRD
ogmmf_vrd = OGMMF_VRD()
ogmmf_vrd.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
ogmmf_vrd.plotAccuracy()
print("Accuracy: ", ogmmf_vrd.accuracyGeneral())

# Ensemble Learners
ensembles = Ensemble_Learners(ensemble="LevBag")
ensembles.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
ensembles.plotAccuracy()
print("Accuracy: ", ensembles.accuracyGeneral())

# HAT + DD
hat_adwin = Hoeffding_Detector(drift_detector="ADWIN")
hat_adwin.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
hat_adwin.plotAccuracy()
print("Accuracy: ", hat_adwin.accuracyGeneral())

# GMM-VRD
gmm_vrd = GMM_VRD()
gmm_vrd.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
gmm_vrd.plotAccuracy()
print("Accuracy: ", gmm_vrd.accuracyGeneral())

# Dynse
dynse = Dynse()
dynse.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
dynse.plotAccuracy()
print("Accuracy: ", dynse.accuracyGeneral())

# IGMM-CD
igmm_cd = IGMM_CD()
igmm_cd.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
igmm_cd.plotAccuracy()
print("Accuracy: ", igmm_cd.accuracyGeneral())
####### 2. DEFINING THE MODELS ####################################################################
    
    
####### 3. STORING THE PREDICTIONS ################################################################
import pandas as pd
df = pd.DataFrame(data={'predictions': ogmmf_vrd.PREDICTIONS, 'target':ogmmf_vrd.TARGET})
df.to_csv("/images/"+ogmmf_vrd.NAME+"-"+dataset[i]+".csv")
####### 3. STORING THE PREDICTIONS ################################################################


