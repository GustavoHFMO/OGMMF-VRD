'''
Created on 11 de nov de 2019
@author: gusta
'''
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
al = Adjust_labels()
import numpy as np

def write_to_arff(stream, name, name_save, unique, output_path):
        '''
        method to write arff file
        :param: output_path: location to store the data
        '''
        
        classes = ""
        if(name=="virtual_5changes"):
            classes = "{0,1,2}"
        elif(name=="virtual_9changes"):
            classes = "{0,1,2}"
        
        
        arff_writer = open(output_path+name_save+".arff", "w")
        arff_writer.write("@relation " + name_save + "\n")
        arff_writer.write("@attribute x real" + "\n" +
                              "@attribute y real" + "\n" +
                              "@attribute class "+classes+ "\n\n")
        arff_writer.write("@data" + "\n")
        for i in range(0, len(stream)):
            arff_writer.write(str("%0.3f" % stream[i][0]) + "," +
                              str("%0.3f" % stream[i][1]) + "," +
                              str("%d" % stream[i][2]) + "\n")
            
            
        arff_writer.close()
        print("You can find the generated files in " + output_path + name_save + "!")

def create_noisy(labels, tx):
    
    new_labels = labels
    
    def generateNewLabel(unique, label):
        
        # removing the true class
        rest = unique
        rest = np.delete(rest, int(label))
        
        # generate random class
        generated = np.random.choice(rest)
        
        # returning the new class
        return generated
    
    # knowing the number of classes
    
    unique, _ = np.unique(labels, return_counts=True)
    
    # defining the number of examples that will be noisy
    qtd = int(np.floor(tx*len(labels))) 
    
    # change the labels 
    for i in range(qtd):
        print(i)
        # element that will be changed
        element = np.random.randint(len(labels))
        
        # generating a new label
        new_labels[element] = generateNewLabel(unique, new_labels[element])
    
    # returning the new labels
    return new_labels

    
# datasets and variations
#dataset = ['circles', 'sine2', 'virtual_9changes']
#noisy = [0.15, 0.2, 0.25, 0.3]

dataset = ['virtual_5changes']
noisy = [0.05, 0.1, 0.15, 0.2]

# reading the dataset
for i in range(len(dataset)):
    labels, _, stream_records = ARFFReader.read("../../data_streams/_synthetic/"+dataset[i]+".arff")
    stream_records = al.adjustStream(labels, stream_records)
        
    for j in noisy:
        # generating noisy
        stream_records[:,-1] = create_noisy(stream_records[:,-1], j)
        unique, _ = np.unique(stream_records[:,-1], return_counts=True)
        # writing the new stream
        write_to_arff(stream_records, dataset[i], dataset[i]+"-noisy="+str(j), unique, "../../data_streams/noisy/")

