"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import random
from streams.generators.tools.transition_functions import Transition

class CIRCLES:

    def __init__(self, concept_length=25000, transition_length=500, noise_rate=0.1, random_seed=10):
        self.__CIRCLES = [[[0.2, 0.5], 0.15], [[0.4, 0.5], 0.2], [[0.6, 0.5], 0.25], [[0.8, 0.5], 0.3]]
        self.PARAMS = self.__CIRCLES
        self.__INSTANCES_NUM = concept_length * len(self.__CIRCLES)
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_DRIFTS = len(self.__CIRCLES) - 1
        self.__W = transition_length
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
        return CIRCLES.__name__

    def generate(self, output_path="CIRCLES"):

        random.seed(self.__RANDOM_SEED)

        # [1] CREATING RECORDS - criar todas as mudancas ao longo do stream
        # for para iterar sobre o numero maximo de instancias por conceito
        for i in range(0, self.__INSTANCES_NUM):
            # iteracao comum
            concept_sec = int(i / self.__CONCEPT_LENGTH)
            # escolher os parametros do respectivo conceito
            record = self.create_record(self.__CIRCLES[concept_sec])
            # salvar o padrao
            self.__RECORDS.append(list(record))

        # [2] TRANSITION - criar a transicao do conceito antigo para o conceito novo
        for i in range(0, self.__NUM_DRIFTS):
            
            # variavel para salvar as transicoes/mudancas
            transition = []
            
            # cria um tipo de perturbacao para nao gerar apenas dados de um so parametro, cria de dois
            for j in range(0, self.__W):
                if random.random() < Transition.sigmoid(j, self.__W):
                    record = self.create_record(self.__CIRCLES[i + 1])
                else:
                    record = self.create_record(self.__CIRCLES[i])
                transition.append(list(record))
                
            # definindo o inicio e o fim do conceito    
            starting_index = i * self.__CONCEPT_LENGTH
            ending_index = starting_index + self.__W
            
            # substituindo no vetor geral as transicoes/mudancas geradas
            self.__RECORDS[starting_index: ending_index] = transition

        # [3] ADDING NOISE - adicionando ruido aos dados
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        # escrevendo a stream
        self.write_to_arff(output_path + ".arff")

    def create_record(self, circle):
        '''
        :circle: parametros de centro e raio
        '''
        
        # gerando uma obsevacao
        x, y, c = self.create_attribute_values(circle)
        
        # gerando observacoes das duas classes de forma aleatoria
        if random.random() < 0.5:
            while c == 'p':
                x, y, c = self.create_attribute_values(circle)
        else:
            while c == 'n':
                x, y, c = self.create_attribute_values(circle)
        
        # retornando o padrao
        return x, y, c

    def create_attribute_values(self, circle):
        '''
        :circle: parametros de centro e raio
        '''
        
        # geracao de um padrao aleatorio
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # pegando o resultado de uma observacao gerada pela equacao do circulo
        res = self.get_circle_result(circle[0], x, y, circle[1])
        
        # se o resultado for maior que zero vai ser da classe p, caso contrario n
        c = 'p' if res > 0 else 'n'
        
        # retorna o padrao e a classe
        return x, y, c

    @staticmethod
    def get_circle_result(c, x, y, radius):
        '''
        :c: referente ao centro do circulo
        :x: valor para eixo de x
        :y: valor para eixo de y
        :radius: raio de geracao do circulo
        '''
        
        # equacao do circulo
        return (x - c[0])**2 + (y - c[1])**2 - radius**2
    
    @staticmethod
    def equation(c, x, y, radius):
        '''
        :c: referente ao centro do circulo
        :x: valor para eixo de x
        :y: valor para eixo de y
        :radius: raio de geracao do circulo
        '''
        
        # equacao do circulo
        res = (x - c[0])**2 + (y - c[1])**2 - radius**2
        
        # retornando a classe para a equacao
        return 'p' if res > 0 else 'n' 

    def add_noise(self):
        for i in range(0, len(self.__NOISE_LOCATIONS)):
            noise_spot = self.__NOISE_LOCATIONS[i]
            c = self.__RECORDS[noise_spot][2]
            if c == 'p':
                self.__RECORDS[noise_spot][2] = 'n'
            else:
                self.__RECORDS[noise_spot][2] = 'p'

    def write_to_arff(self, output_path):
        arff_writer = open(output_path, "w")
        arff_writer.write("@relation __CIRCLES" + "\n")
        arff_writer.write("@attribute x real" + "\n" +
                          "@attribute y real" + "\n" +
                          "@attribute class {p,n}" + "\n\n")
        arff_writer.write("@data" + "\n")

        for i in range(0, len(self.__RECORDS)):
            arff_writer.write(str("%0.5f" % self.__RECORDS[i][0]) + "," +
                              str("%0.5f" % self.__RECORDS[i][1]) + "," +
                              self.__RECORDS[i][2] + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")

def main():

    def caminho(nome, i):
        import os
        stream_name = nome
        project_path = "../../data_streams/_synthetic/noisy/" + stream_name + "/"
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        file_path = project_path + stream_name + "_"+ str(i)
        
        return file_path
    
    
    # parametros 
    tam_conceito = 2000
    
    # gerando o dataset com as respectivas configuracoes
    file_path = caminho('AAAA', 1)
    stream_generator = CIRCLES(concept_length=tam_conceito, random_seed=1)
    stream_generator.generate(file_path)
    
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
            classe = stream_generator.equation(params[0], x[0], x[1], params[1])
            
            # comparando a classe
            if(classe != y):
                diferencas += 1
        
        # calcular as severidades
        severidades.append((100*diferencas)/tam_conceito)
        
    # printar as severidades
    print(severidades)
        
if __name__ == "__main__":
    main() 


