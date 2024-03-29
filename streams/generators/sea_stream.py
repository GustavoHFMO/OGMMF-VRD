"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import random

from streams.generators.tools.transition_functions import Transition

class SEA:

    def __init__(self, concept_length=25000, thresholds=[8, 9, 7, 9.5], transition_length=50,
                 noise_rate=0.1, random_seed=1):

        self.__INSTANCES_NUM = concept_length * len(thresholds)
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_DRIFTS = len(thresholds) - 1
        self.__W = transition_length
        self.__THRESHOLDS = thresholds
        self.PARAMS = self.__THRESHOLDS
        self.__RECORDS = []
        self.RECORDS = self.__RECORDS

        self.__RANDOM_SEED = random_seed
        random.seed(self.__RANDOM_SEED)
        self.__NOISE_LOCATIONS = random.sample(range(0, self.__INSTANCES_NUM), int(self.__INSTANCES_NUM * noise_rate))

        print("You are going to generate a " + self.get_class_name() + " data stream containing " +
              str(self.__INSTANCES_NUM) + " instances, and " + str(self.__NUM_DRIFTS) + " concept drifts; " + "\n\r" +
              "where they appear at every " + str(self.__CONCEPT_LENGTH) + " instances.")

    @staticmethod
    def get_class_name():
        return SEA.__name__

    def generate(self, output_path="SEA"):

        random.seed(self.__RANDOM_SEED)

        # [1] CREATING RECORDS
        for i in range(0, self.__INSTANCES_NUM):
            concept_sec = int(i / self.__CONCEPT_LENGTH)
            record = self.create_record(self.__THRESHOLDS[concept_sec])
            self.__RECORDS.append(list(record))

        # [2] TRANSITION
        for i in range(0, self.__NUM_DRIFTS):
            transition = []
            for j in range(0, self.__W):
                if random.random() < Transition.sigmoid(j, self.__W):
                    record = self.create_record(self.__THRESHOLDS[i + 1])
                else:
                    record = self.create_record(self.__THRESHOLDS[i])
                transition.append(list(record))
            starting_index = i * self.__CONCEPT_LENGTH
            ending_index = starting_index + self.__W
            self.__RECORDS[starting_index: ending_index] = transition

        # [3] ADDING NOISE
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        self.write_to_arff(output_path + ".arff")

    def create_record(self, dist_id):
        x, y, z, c = self.create_attribute_values(dist_id)
        if random.random() < 0.5:
            while c != 'p':
                x, y, z, c = self.create_attribute_values(dist_id)
        else:
            while c != 'n':
                x, y, z, c = self.create_attribute_values(dist_id)
        return x, y, z, c

    @staticmethod
    def create_attribute_values(threshold):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        z = random.uniform(0, 10)
        c = 'p' if x + y <= threshold else 'n'
        return x, y, z, c
    
    @staticmethod
    def equation(x, y, threshold):
        '''
        :x: valor para eixo de x
        :y: valor para eixo de y
        '''
        
        # retornando a classe para a equacao
        return 'p' if x + y <= threshold else 'n' 
    
    def add_noise(self):
        for i in range(0, len(self.__NOISE_LOCATIONS)):
            noise_spot = self.__NOISE_LOCATIONS[i]
            c = self.__RECORDS[noise_spot][3]
            if c == 'p':
                self.__RECORDS[noise_spot][3] = 'n'
            else:
                self.__RECORDS[noise_spot][3] = 'p'

    def write_to_arff(self, output_path):
        arff_writer = open(output_path, "w")
        arff_writer.write("@relation SEA" + "\n")
        arff_writer.write("@attribute x real" + "\n" +
                          "@attribute y real" + "\n" +
                          "@attribute z real" + "\n" +
                          "@attribute class {n,p}" + "\n\n")
        arff_writer.write("@data" + "\n")

        for record in self.__RECORDS:
            arff_writer.write(str("%0.3f" % record[0]) + "," + str("%0.3f" % record[1]) + "," +
                              str("%0.3f" % record[2]) + "," + record[3] + "\n")

        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")

def main():

    def caminho(nome, i):
        import os
        stream_name = nome
        #project_path = "../../data_streams/_synthetic/"
        #project_path = "../../data_streams/noisy/"
        project_path = "../../data_streams/noisy/"
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        #file_path = project_path + stream_name + "_"+ str(i)
        file_path = project_path + stream_name
        
        return file_path
    
    # parametros 
    tam_conceito = 2000
    
    # gerando o dataset com as respectivas configuracoes
    file_path = caminho("SEARec-noisy=0.2", 1)
    #stream_generator = SEA(concept_length=tam_conceito, thresholds=[1, 9, 2, 6], noise_rate=0.2, random_seed=1)
    stream_generator = SEA(concept_length=tam_conceito, thresholds=[1, 9, 2, 6, 1, 9, 2, 6], noise_rate=0.2, random_seed=1)
    stream_generator.generate(file_path)
    
    '''
    # stream gerado
    STREAM = stream_generator.RECORDS
    
    # qtd de conceitos
    qtd_conceitos = int(len(STREAM)/tam_conceito)
    
    # salvar as severidades 
    severidades = []
    
    # iterar sobre os conceitos
    for i in range(1, qtd_conceitos):
        conceito_velho = STREAM[tam_conceito*i-1:tam_conceito*(i)]
        conceito_novo = STREAM[tam_conceito*(i):tam_conceito*(i+1)]
        
        # pegando os parametros do conceito passado
        params = stream_generator.PARAMS[i-1]
    
        # variavel para salvar as diferencas
        diferencas = 0
            
        # passando os dados do conceito novo na equacao
        for OBS in conceito_novo:
            # observacao
            x = OBS[0:-1]
            y = OBS[-1]
            # equacao do conceito passado
            classe = stream_generator.equation(x[0], x[1], params)
            
            # comparando a classe
            if(classe != y):
                diferencas += 1
        
        # calcular as severidades
        severidades.append((100*diferencas)/tam_conceito)
        
    # printar as severidades
    print(severidades)
    '''

if __name__ == "__main__":
    main()