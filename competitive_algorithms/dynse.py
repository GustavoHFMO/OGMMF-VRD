'''
Created on 22 de set de 2018
@author: gusta
Detalhe importante sobre o dynse, no servidor ele deve ser executado fora do anaconda e com o python3.
Se for executado diferente disso tera resultados diferentes.
'''

# Importing dynamic selection techniques:
from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.dcs.lca import LCA
from deslib.dcs.ola import OLA 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
al = Adjust_labels()
#import pandas as pd
import numpy as np
import warnings
import copy
import time
plt.style.use('seaborn-whitegrid')
warnings.filterwarnings("ignore")

class PrunningEngine:
    def __init__(self, Type):
        '''
        classe para instanciar o tipo de poda do dynse
        :param: type: tipo da poda [age, accuracy]
        '''
        self.TYPE = Type

    def prunning(self, P, W, C, D):
        '''
        metodo para podar a quantidade de classificadores
        :param: P: pool de classificadores
        :param: W: janela com as instancias a serem avaliadas
        :param: C: novo classificador a ser adicionado
        :param: D: tamanho maximo do pool
        '''
        
        if(self.TYPE=='age'):
            return self.ageBased(P, W, C, D)
        elif(self.TYPE=='accuracy'):
            return self.accuracyBased(P, W, C, D)
    
    def ageBased(self, P, W, C, D):
        '''
        metodo para podar a quantidade de classificadores baseado no classificador mais antigo
        :param: P: pool de classificadores
        :param: W: janela com as instancias a serem avaliadas
        :param: C: novo classificador a ser adicionado
        :param: D: tamanho maximo do pool
        '''
            
        # adicionando um novo classificador ao pool
        P.append(C)
        
        # removendo o classificador mais antigo
        if(len(P)> D):
            del P[0]
                
        return P
    
    def accuracyBased(self, P, W, C, D):
        '''
        metodo para podar a quantidade de classificadores baseado no classificador com menor desempenho
        :param: P: pool de classificadores
        :param: W: janela com as instancias a serem avaliadas
        :param: C: novo classificador a ser adicionado
        :param: D: tamanho maximo do pool
        '''

        # adicionando um novo classificador ao pool
        P.append(C)
        
        # processo para remover o classificador
        if(len(P)> D):
                    
            # ajustando a janela de validacao
            new_W = W[0]
            for i in range(1, len(W)):
                new_W = np.concatenate((new_W, W[i]), axis=0)
            
            # dados para verificar a acuracia dos modelos
            x = new_W[:,0:-1]
            y = new_W[:,-1]
            
            # computando a acuracia de todos os modelos em W
            acuracia = []
            for classificador in P:
                y_pred = classificador.predict(x)
                acuracia.append(accuracy_score(y, y_pred))
                
            # excluindo o classificador com pior desempenho
            del P[np.argmin(acuracia)]
                
        return P

class ClassificationEngine:
    def __init__(self, Type):
        '''
        classe para instanciar o tipo de mecanismo de classificacao do dynse
        :param: type: tipo da poda ['knorae', 'knorau', 'ola', 'lca', 'posteriori', 'priori']
        '''
        self.TYPE = Type
        
    def fit(self, x_sel, y_sel, P, k):
        '''
        metodo para chamar o tipo de DS
        :param: x_sel: dados de treinamento da janela de validacao
        :param: y_sel: rotulos da janela de validacao
        :param: P: pool de classificadores
        :param: k: vizinhanca
        '''
        
        # escolhendo a tecnica de selecao de classificadores
        if(self.TYPE=='knorae'):
            DS = KNORAE(P, k)
        elif(self.TYPE=='knorau'):
            DS = KNORAU(P, k)
        elif(self.TYPE=='ola'):
            DS = OLA(P, k)
        elif(self.TYPE=='lca'):
            DS = LCA(P, k)
        elif(self.TYPE=='posteriori'):
            DS = APosteriori(P, k)
        elif(self.TYPE=='priori'):
            DS = APriori(P, k)
            
        # encontrando os classificadores competentes do DS escolhido
        self.DS = copy.deepcopy(DS)           
        self.DS.fit(x_sel, y_sel)
        
    def predict(self, x):
        '''
        metodo para realizar a predicao com o tipo de classificador selecionado
        :param: x: variaveis de entrada    
        :return: labels referentes a entrada x
        '''
        
        return self.DS.predict(x)
    
class Dynse(PREQUENTIAL_SUPER):
    def __init__(self, D=10, M=100, K=5, train_size=50, window_size=200):
        '''
        Dynamic Selection Based Drift Handler Framework
        :param: D: tamanho maximo do pool
        :param: M: tamanho da janela de estimacao de acuracia
        :param: K: tamanho da vizinhanca
        :param: CE: mecanismo de classificacao
        :param: PE: mecanismo de poda
        :param: BC: classificador base
        '''
        
        self.start = window_size
        
        self.D = D
        self.M = M
        self.K = K
        self.CE = ClassificationEngine('priori')
        self.PE = PrunningEngine('age')
        self.BC = GaussianNB()
        self.train_size = train_size
        
        self.PREDICTIONS = []
        self.TARGET = []
        #self.NAME = "Dynse-"+self.CE.TYPE+"-"+self.PE.TYPE
        self.NAME = "Dynse"
        
        # variable to count the cross validation
        self.count = 0
    
    def adjustingWindowBatch(self, W):
        '''
        metodo para ajustar a janela de validacao
        :param: W: janela de validacao
        '''
        
        # ajustando a janela de validacao
        new_W = W[0]
        for i in range(1, len(W)):
            new_W = np.concatenate((new_W, W[i]), axis=0)
        
        # dados para treinar
        x = new_W[:,0:-1]
        y = new_W[:,-1]
        
        # retornando os dados
        return x, y
    
    def adjustingWindowOne(self, W):
        '''
        metodo para ajustar a janela de validacao
        :param: W: janela de validacao
        '''
        
        # ajustando a janela de validacao
        new_W = np.asarray(W)
        
        # dados para treinar
        x = new_W[:,0:-1]
        y = new_W[:,-1]
        
        # retornando os dados
        return x, y
     
    def dividingPatternLabel(self, B):
        '''
        metodo para dividir os dados do batch em treinamento e Exp_cfc
        :param: B: batch a ser dividido
        :param: batch_train: tamanho do batch para treinamento 
        '''
        
        B = np.asarray(B)
        x, y = B[:, 0:-1], B[:,-1]
        
        return x, y
    
    def trainNewClassifier(self, BC, B_train):
        '''
        metodo para treinar um classificador 
        :param: BC: classificador base a ser utilizado
        :param: B: batch a ser treinado
        '''
        
        #obtendo os dados para treinamento e o de Exp_cfc
        x, y = self.dividingPatternLabel(B_train)

        # fazendo uma copia do classe do classificador
        C = copy.deepcopy(BC)
        
        # treinando o classificador
        C.fit(x, y)
        
        # retornando
        return C
        
    def removeOldestBatch(self, W):
        '''
        metodo para remover o batch mais antigo
        :param: W: janela que ira remover o mais antigo
        '''
        
        # removing the last element
        #del W[self.number_labels]
        if(len(W) > self.M):
            del W[0]
        
        # returning the window
        return W

    def removeOldestBatchBug(self, W):
        '''
        metodo para remover o batch mais antigo
        :param: W: janela que ira remover o mais antigo
        '''
        
        # removing the last element
        if(len(W) > self.M):
            del W[self.number_labels]
        
        # returning the window
        return W
        
    def prequential_batch(self, labels, stream, step_size, train_size):
        '''
        metodo para executar o codigo
        :param: labels: rotulos existentes no stream
        :param: stream: fluxo de dados
        :param: batch_size: tamanho dos batches
        '''

        # salvando o stream e o tamanho do batch
        self.STREAM = al.adjustStream(labels, stream)
        
        # janela inicial
        W = []
        
        # pool inicial de classificadores
        P = []
        
        # for para percorrer a stream
        for i in range(0, len(self.STREAM), step_size):
            
            # obtendo o atual batch
            B = self.STREAM[i:i+step_size]

            # Etapa com dados rotulados ##############################
            
            # obtendo os dados rotulados
            B_train = B[:train_size]
            
            # adicionando o batch na janela
            W.append(B_train)
            
            # treinando um classificador 
            C = self.trainNewClassifier(self.BC, B_train)
                    
            # podando o numero de classificadores
            P = self.PE.prunning(P, W, C, self.D)
                
            # verificando o tamanho da janela
            if(len(W) > self.M):
    
                # removendo o batch mais antigo 
                self.removeOldestBatch(W)

            
            # Etapa com dados nao rotulados ###########################
                    
            # obtendo os dados nao rotulados
            B_test = B[train_size:]
            
            # ajustando a janela de validacao
            x_sel, y_sel = self.adjustingWindowBatch(W)
                    
            # ajustando o mecanismo de classificacao
            self.CE.fit(x_sel, y_sel, P, self.K)
                
            # realizando a classificacao de cada instancia em B
            for x in B_test:
                    
                # recebendo o atual padrao e o seu rotulo
                pattern, label = np.asarray([x[0:-1]]), x[-1]
                    
                # realizando a classificacao
                y_pred = self.CE.predict(pattern)
                
                # salvando a previsao e o alvo
                self.PREDICTIONS.append(y_pred[0])
                self.TARGET.append(label)
                
            # printando a execucao
            self.printIterative(i)
    
    def plotLearnedBoundaries(self, model, X, Y, show=False):
        
        def plotData(x, y):
            # defining the colors of each gaussian
            unique, _ = np.unique(y, return_counts=True)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique)))
            marks = ["^", "o", '+', ',']
            
            # creating the image
            plt.subplot(111)
                
            # receiving each observation per class
            classes = []
            for i in unique:
                aux = []
                for j in range(len(y)):
                    if(y[j] == i):
                        aux.append(x[j])
                classes.append(np.asarray(aux))
            classes = np.asarray(classes)
            
            # plotting each class
            for i in unique:
                i = int(i)
                plt.scatter(classes[i][:,0],
                            classes[i][:,1],
                            color = colors[i],
                            marker = marks[i])
                            #label = 'class '+str(i))
            
        def make_meshgrid(x, y, h=.01):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            return xx, yy
        
        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, **params)
            ax.contour(xx, yy, Z,
                    linewidths=3, 
                    linestyles='solid',
                    cmap=plt.cm.rainbow, 
                    zorder=0)
            # adding the label of true boundaries
            c = plt.cm.rainbow(np.linspace(0, 1, 3))
            #ax.plot([],[], linewidth=2, linestyle='solid', color=c[0], label="Learned Boundaries")
            
        # plot data
        plotData(X, Y)
        # creating the boundaries
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(plt, model, xx, yy, cmap=plt.cm.rainbow, alpha=0.1)
        
        # showing
        ad = 0.2
        plt.axis([X0.min()-ad, X0.max()+ad, X1.min()-ad, X1.max()+ad])
        
        if(show):
            plt.show()
            
    def plotTrueBoundaries(self, X, Y):
        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            return xx, yy
        
        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            #ax.contourf(xx, yy, Z, **params)
            ax.contour(xx, yy, Z,
                    linewidths=2, 
                    linestyles=':',
                    colors='dimgray', zorder=0)
            # adding the label of true boundaries
            
        # training a SVM over the data
        from sklearn import svm
        svm_model = svm.SVC(kernel='rbf', gamma=2, C=2)
        svm_model.fit(X, Y)
        
        # defining the true boundaries
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(plt, svm_model, xx, yy, cmap=plt.cm.rainbow, alpha=0.1, zorder=0)
        
        # adding the legend
        clr = plt.cm.rainbow(np.linspace(0, 1, 3))
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], linewidth=3, linestyle=':', 
                                  color='dimgray', label="True Boundaries"),
                           Patch(facecolor=clr[0], edgecolor=clr[0], alpha=0.5,
                                 label='Class 0: Learned Boundaries'),
                           Patch(facecolor=clr[1], edgecolor=clr[1], alpha=0.5,
                                 label='Class 1: Learned Boundaries'),
                           Patch(facecolor=clr[2], edgecolor=clr[2], alpha=0.5,
                                 label='Class 2: Learned Boundaries')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # adjusting the margins of the plot
        plt.subplots_adjust(left=0.04, bottom=0.04, right=0.99, top=0.99)
        plt.show()
        
    def creatingDataBug(self, stream, labels):
        '''
        creating artificial data for all classes
        '''
        
        # storing the number of labels
        labels = np.unique(stream[:,-1])
        self.number_labels = len(labels)
        
        # creating random patterns
        #patterns = np.random.rand(self.number_labels, len(stream[0]))
        patterns = np.zeros((self.number_labels, len(stream[0])))
        patterns[:,-1] = np.asarray(labels, dtype='float')
        
        # using these data to avoid problems
        self.patterns_problem = patterns
        
        # returning the artificial data
        return list(patterns)
    
    def joiningBatches(self, L):
        '''
        Joining the artificial instances with the currents 
        '''

        # joining
        for i in self.patterns_problem:
            L.append(i)

        # returning            
        return L
    
    def run2(self, labels, stream, cross_validation=True, fold=0, qtd_folds=30):
        '''
        metodo para executar o codigo
        :param: labels: rotulos existentes no stream
        :param: stream: fluxo de dados
        :param: batch_size: tamanho dos batches
        '''

        # starting time
        self.start_time = time.time()

        # storing the new stream
        self.STREAM = stream
        
        # to adjust the plot
        #self.STREAM_Plot = self.STREAM
        #self.STREAM = self.STREAM[1800:]
        
        # janela inicial
        #W = self.creatingDataBug(self.STREAM)
        W = []
        
        # pool inicial de classificadores
        P = []
    
        # variable to store patterns for train
        L = []
        
        # for para percorrer a stream
        for i, X in enumerate(self.STREAM):

            # to use the cross validation
            run=True
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = False
            
            # to execute the prequential precedure
            if(run):
                
                # split the current example on pattern and label
                x, y = X[0:-1], int(X[-1])
                
                # storing the patterns
                L.append(X)
                
                # working to fill the window
                W.append(X)
                    
                # working with full window
                if(i >= self.train_size):
                    
                    # ajustando a janela de validacao
                    x_sel, y_sel = self.adjustingWindowOne(W)
                    
                    # ajustando o mecanismo de classificacao
                    self.CE.fit(x_sel, y_sel, P, self.K)

                    # realizando a classificacao
                    y_pred = self.CE.predict(np.asarray([x]))

                    # guardar a predicao                    
                    if(i >= self.start):
                        self.PREDICTIONS.append(y_pred[0])
                        self.TARGET.append(y)
                        
                    # training a new classifier
                    if(len(L) > self.train_size):

                        # treinando um classificador 
                        #L = np.asarray(self.joiningBatches(L))
                        C = self.trainNewClassifier(self.BC, np.asarray(L))
                        # erasing patterns
                        L = []
                        # podando o numero de classificadores
                        P = self.PE.prunning(P, W, C, self.D)
                       
                    '''
                    ################### to plot #######################################                      
                    #print(i)
                    if(i<200):
                        #x, y = self.dividingPatternLabel(np.asarray(L))
                        #self.plotLearnedBoundaries(C, x, y)
                        #concept = self.STREAM_Plot[0:2000]
                        #self.plotTrueBoundaries(concept[:,0:-1], concept[:,-1])
                        print()
                    elif(i>250):
                        x, y = self.dividingPatternLabel(np.asarray(W))
                        self.plotLearnedBoundaries(P[-1], x, y)
                        concept = self.STREAM_Plot[2001:4000]
                        self.plotTrueBoundaries(concept[:,0:-1], concept[:,-1])
                    ###################################################################
                    '''
                    
                else:
                    # treinando um classificador 
                    C = self.trainNewClassifier(self.BC, np.asarray(L))
                    # erasing patterns
                    L = []
                    # podando o numero de classificadores
                    P = self.PE.prunning(P, W, C, self.D)    
                    
                # removendo o batch mais antigo 
                W = self.removeOldestBatch(W)
                print(len(W))
                
                # printando a execucao
                self.printIterative(i+self.start)

        # end time
        self.end_time = time.time()

    def run(self, labels, stream, cross_validation=True, fold=0, qtd_folds=30):
        '''
        metodo para executar o codigo
        :param: labels: rotulos existentes no stream
        :param: stream: fluxo de dados
        :param: batch_size: tamanho dos batches
        '''

        # starting time
        self.start_time = time.time()

        # storing the new stream
        self.STREAM = stream
        
        # to adjust the plot
        #self.STREAM_Plot = self.STREAM
        #self.STREAM = self.STREAM[1800:]
        
        # janela inicial
        W = self.creatingDataBug(stream, labels)
        #W = []
        
        # pool inicial de classificadores
        P = []
    
        # variable to store patterns for train
        L = []
        
        # for para percorrer a stream
        for i, X in enumerate(self.STREAM):

            # to use the cross validation
            run=False
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = True
            
            # to execute the prequential precedure
            if(run):
                
                # split the current example on pattern and label
                x, y = X[0:-1], int(X[-1])
                
                # storing the patterns
                L.append(X)
                
                # working to fill the window
                W.append(X)
                    
                # working with full window
                if(i >= self.train_size):
                    
                    # ajustando a janela de validacao
                    x_sel, y_sel = self.adjustingWindowOne(W)
                    
                    # ajustando o mecanismo de classificacao
                    self.CE.fit(x_sel, y_sel, P, self.K)

                    # realizando a classificacao
                    y_pred = self.CE.predict(np.asarray([x]))

                    # guardar a predicao                    
                    if(i >= self.start):
                        self.PREDICTIONS.append(y_pred[0])
                        self.TARGET.append(y)
                        
                    # training a new classifier
                    if(len(L) > self.train_size):

                        # treinando um classificador 
                        L = np.asarray(self.joiningBatches(L))
                        # ajustar linha 249
                        #C = self.trainNewClassifier(self.BC, np.asarray(L))
                        C = self.trainNewClassifier(self.BC, L)
                        # erasing patterns
                        L = []
                        # podando o numero de classificadores
                        P = self.PE.prunning(P, W, C, self.D)
                       
                    '''
                    ################### to plot #######################################                      
                    #print(i)
                    if(i<200):
                        #x, y = self.dividingPatternLabel(np.asarray(L))
                        #self.plotLearnedBoundaries(C, x, y)
                        #concept = self.STREAM_Plot[0:2000]
                        #self.plotTrueBoundaries(concept[:,0:-1], concept[:,-1])
                        print()
                    elif(i>250):
                        x, y = self.dividingPatternLabel(np.asarray(W))
                        self.plotLearnedBoundaries(P[-1], x, y)
                        concept = self.STREAM_Plot[2001:4000]
                        self.plotTrueBoundaries(concept[:,0:-1], concept[:,-1])
                    ###################################################################
                    '''
                    
                    # removendo o batch mais antigo 
                    W = self.removeOldestBatchBug(W)
                    
                else:
                    # treinando um classificador 
                    C = self.trainNewClassifier(self.BC, np.asarray(L))
                    # erasing patterns
                    L = []
                    # podando o numero de classificadores
                    P = self.PE.prunning(P, W, C, self.D)    
                    
                # printando a execucao
                self.printIterative(i+self.start)

        # end time
        self.end_time = time.time()

def main():
    
    #np.random.seed(0)
    
    #1. importando o dataset
    i = 0
    dataset = ['gassensor', 'INSECTS-gradual_balanced_norm', 'INSECTS-incremental-abrupt_balanced_norm']
    labels, _, stream_records = ARFFReader.read("../data_streams/usp/"+dataset[i]+".arff")
    #i = 2
    #dataset = ['PAKDD', 'elec', 'noaa', 'poker-lsn2', 'covtypeNorm2']
    #labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    #dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    #labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    #labels, _, stream_records = ARFFReader.read("../data_streams/noisy/sine2-noisy=0.1.arff")
    stream_records = al.adjustStream(labels, stream_records)
    #stream_records = stream_records[:3000]
    
    dynse = Dynse()
    dynse.run2(labels, stream_records, cross_validation=True, fold=2, qtd_folds=10)
    dynse.plotAccuracy()
    
    # printando a acuracia final do sistema
    print(dynse.accuracyGeneral())
    
    
    # salvando a predicao do sistema
    #df = pd.DataFrame(data={'predictions': dynse.PREDICTIONS})
    #df.to_csv("../projects/"+dynse.NAME+"-"+dataset[i]+".csv")
        
if __name__ == "__main__":
    main()        

