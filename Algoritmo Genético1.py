# -*- coding: utf-8 -*-
"""
@author: rzibordi
"""
import os
import pandas as pd
import numpy as np
os.chdir('c:/ricardo')

df = pd.read_csv('Teste_AG.csv', sep=';', decimal=',') #carregando o arquivo de dados

# Separando as variáveis entre X's e Y
x = df[['x1', 'x2', 'x3']]
y = df['y']

# Construindo o modelo de Machine Learning
from sklearn.model_selection import train_test_split 
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.2, random_state=11)

from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor(n_estimators=3000, 
                               random_state=11,
                               min_samples_split=2,
                               max_depth=None,
                               min_samples_leaf=2,
                               max_features='auto',
                               n_jobs=-1)

# Treinando o modelo
modelo.fit(x_treino,y_treino)

# Chamando a biblioteca de Algoritmo Genético no Python
import pygad as ga
objetivo = 200  # Valor que estimo ser um valor ótimo

# Criando a função que calcula 
def fitness_func(solution, solution_idx): # Essa função é inerente ao pacote pygad
    output = float(modelo.predict([np.array(solution)]))
    fitness = 1.0 / np.abs(output - objetivo)
    return fitness

# Parâmetros do Algoritmo Genético
fitness_function = fitness_func
num_generations = 80
num_parents_mating = 4
sol_per_pop = 8
num_genes = 3 # número de respostas
init_range_low = -2
init_range_high = 5
parent_selection_type = "sss"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 15

# Construindo o modelo que irá gerar a população inicial e gerar as aleatoriedades 
# nas combinações das soluções descendentes, com mutações e crossing-over
ga_instance = ga.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_percent_genes=mutation_percent_genes)

# Executando o modelo
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Melhores soluções : {solution}".format(solution=solution))
print("Valor da função fitness da solução = {solution_fitness}".format(solution_fitness=solution_fitness))
prediction = float(modelo.predict([np.array(solution)]))
print("Valor ótimo calculado pelo modelo : {prediction}".format(prediction=prediction))
