'''
Created on 16 de jan de 2019
@author: gusta
'''

class SUPER_DETECTOR():
    def __init__(self):
        pass

    def fit(self, classifier, W):
        '''
        method to fit to the current concept 
        '''
        
        for obs in W:
            x, y = obs[:-1], obs[-1]
            
            try:
                yi = classifier.predict(x)
            except:
                yi = classifier.predict([x])
            
            pred = True
            if(yi != y):
                pred = False
                
            self.run(pred)
        
    def detect(self, y_true, y_pred):
        '''
        method to monitor the index
        '''
        
        # checkint out the prediction of classifier
        pred = True
        if(y_true != y_pred):
            pred = False
        
        warning_level, change_level = self.run(pred)
        
        return warning_level, change_level
    
    
    