from streams.readers.arff_reader import ARFFReader
import matplotlib.pyplot as plt
import numpy as np

i = 4
dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
labels, _, stream_records = ARFFReader.read("../../data_streams/_synthetic/"+dataset[i]+".arff")

stream_records = stream_records[8900:]
stream = np.asarray(stream_records).astype(np.float)

classe = 0
classe_buscada = [stream[x] for x, i in enumerate(stream[:,-1]) if i == classe]
print("Classe: ", classe)
print("media: ", np.mean(classe_buscada, axis=0))
print("desvio: ", np.std(classe_buscada, axis=0))
print("porcentagem: ", 100*len(classe_buscada)/len(stream))

classe = 1
classe_buscada = [stream[x] for x, i in enumerate(stream[:,-1]) if i == classe]
print("Classe: ", classe)
print("media: ", np.mean(classe_buscada, axis=0))
print("desvio: ", np.std(classe_buscada, axis=0))
print("porcentagem: ", 100*len(classe_buscada)/len(stream))

classe = 2
classe_buscada = [stream[x] for x, i in enumerate(stream[:,-1]) if i == classe]
print("Classe: ", classe)
print("media: ", np.mean(classe_buscada, axis=0))
print("desvio: ", np.std(classe_buscada, axis=0))
print("porcentagem: ", 100*len(classe_buscada)/len(stream))

#classe_buscada = np.asarray(classe_buscada)
#plt.scatter(classe_buscada[:,0], classe_buscada[:,1])
#plt.show()
