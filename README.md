# Otm
Otimizando a predição de um Modelo de Machine Learning através de Algoritmos Genéticos
Resolvi postar aqui pois quase não encontrei esse tipo de abordagem nos repositórios usuais de ciências de dados. 

O problema é o seguinte: Há um fenômeno que quero prever, então através dos dados de entrada (X's) e saída (Y) crio um modelo. Após esse modelo criado quero saber quais os melhores X's para resultar no melhor Y !

Após o modelo criado, e aqui, para exemplificar fiz uma modelagem simples com o Random Forest Regressor para modelar  Y = f(x1, x2, x3).

o modelo gerado é a "equação" que será utilizada pelo algoritmo genético para evoluir para a solução ótima, gerando um conjunto de soluções que irá evoluir através de cruzamentos, mutações em gerações de soluções, selecionando as melhores, ou seja, as mais aptas! A convergência é razoável e o tempo de processamento deixou a desejar. Aqui neste exemplo não explorei muito os parâmetros, apenas jogando a ideia para quem precisar!
