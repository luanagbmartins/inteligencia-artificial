import os
import json
import uuid
from sqlalchemy.engine import create_engine
from sqlalchemy import Column, Table, MetaData
from sqlalchemy import Integer, Text, Float
import sqlalchemy

import os
import pandas as pd
import seaborn as sns
import numpy as np  

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

from sklearn.model_selection import train_test_split


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

DATASET = 'dataset.csv'

class DecisionTree:
    """
    Criação da árvore binária de decisão, com uma ramificação para testes 
    Verdadeiros e uma ramificação para testes Falsos.
    """
    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None, summary=None):
        self.col = col                  # Melhor atributo de divisão do nó 
        self.value = value              # Valor utlizado na comparação
        self.trueBranch = trueBranch    # Ramificação de valores Verdadeiros
        self.falseBranch = falseBranch  # Ramificação de valores Falsos
        self.results = results          # None para nós de decisão
        self.summary = summary          # Entropia e tamanho dos dados

from math import log

def divideSet(rows, column, value):

    splittingFunction = None

    # Para valores do tipo int ou float
    if isinstance(value, int) or isinstance(value, float): 
        splittingFunction = lambda row : row[column] >= value
    
    # Para valores do tipo string
    else: 
        splittingFunction = lambda row : row[column] == value

    # Dados que atendem uma dada condição 
    list1 = [row for row in rows if splittingFunction(row)]
    # Dados que não atendem uma daa condição
    list2 = [row for row in rows if not splittingFunction(row)]

    return (list1, list2)


def uniqueCounts(rows):
    
    results = {}
    for row in rows:
        # A resposta da classificação estará na última coluna 
        # Cada resposta diz se o aluno é um bom ou mal aluno
        r = row[-1]
        if r not in results: results[r] = 0
        results[r] += 1

    return results

def entropy(rows):

    log2 = lambda x: log(x) / log(2)        # Função para calcular log2(x)
    results = uniqueCounts(rows)            # Pega todos os resultados e sua respectiva quantidade

    entr = 0.0
    for r in results:
        p = float(results[r]) / len(rows)   # Probabilidade de um dado resultado 
        entr -= p * log2(p)                 # Calcula o grau da entropia dado por 

    return entr

def growDecisionTree(rows):
    """ Gera uma árvore de decisão recursivamente realizando as divisões dos dados dado um atributo de divisão"""

    # Verifica se não há mais dados para serem divididos
    if len(rows) == 0: return DecisionTree()     

    # Realiza o cálculo do grau da entropia do nó pai
    currentScore = entropy(rows)     

    bestGain = 0.0
    bestAttribute = None
    bestSets = None

    # Pega a quantidade de atributos menos a última coluna (coluna de resposta) 
    columnCount = len(rows[0]) - 1              

    for col in range(0, columnCount):
        # Pega todos os valores dos atributos daquele nó
        columnValues = [row[col] for row in rows] 

        # A função set() retorna um conjunto de coleção não ordenada de itens
        # Um conjunto set() é composto por elementos únicos(sem duplicadatas)
        unique = list(set(columnValues))

        # Realiza o teste de escolha de atributo para condição de teste
        for value in unique:
            # Seperação dos dados 
            (set1, set2) = divideSet(rows, col, value)

            p = float(len(set1)) / len(rows)

            # Cálculo do ganho de informação: entropia(pai) - entropia(filhos)
            gain = currentScore - p*entropy(set1) - (1-p)*entropy(set2)

            """ Se o ganho de informação for maior do que já observado
                o atributo em questão se torna o melhor atributo """
            if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                bestGain = gain
                bestAttribute = (col, value)
                bestSets = (set1, set2)

    # Guarda o valor atual da entropia(impureza), e o número de linhas do nó
    summary = {'impurity' : '%.3f' % currentScore, 'samples' : '%d' % len(rows)}

    # Realiza a geração de nós filhos com os melhores sets gerados da divisão
    if bestGain > 0:
        # Ramificação com valores do tipo Verdadeiro
        trueBranch = growDecisionTree(bestSets[0])  

        # Ramificação com valores do tipo Falso
        falseBranch = growDecisionTree(bestSets[1])

        # Armazena o nó pai na árvore
        return DecisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch,
                            falseBranch=falseBranch, summary=summary)

    # Se não há ganho de informação, não há mais ramificação desse nó
    else:
        # Armazena o nó folha na árvore os resultados de classificação
        return DecisionTree(results=uniqueCounts(rows), summary=summary)




class Driver(object):

    def __init__(self):
        self.db = None
        self.engine = None
        self.meta = None

        self.address = 'postgresql+psycopg2://postgres:docker@localhost/postgres?port=5432'
        self.table_files_name = 'benchmarks'
        table_files_args = lambda: (
              Column('id', Text),
              Column('name', Text),
              Column('0,0', Float),
              Column('0,1', Float),
              Column('1,0', Float),
              Column('1,1', Float)
              )
        self.table_files = lambda x: Table(self.table_files_name, x, *table_files_args() )
        self.table_files_query = sqlalchemy.table(self.table_files_name, *table_files_args() )
        

    def connect(self):
        self.db = create_engine(self.address)
        self.engine = self.db.connect()
        self.meta = MetaData(self.engine)
        self.table_files(self.meta)
        self.meta.create_all()
        return self

    def insert(self, params):
        statement = self.table_files_query.insert().values(**params)
        self.engine.execute(statement)
        return self

    def get(self):
        find = self.table_files_query.select()
        return self.engine.execute(find).fetchall()

def prune(tree, minGain):
    """ Realiza a poda da árvore de acordo com um valor mínimo estabelecido."""

    # Olha se os filhos de um nó são folhas(resultado). Se None não é.
    if tree.trueBranch.results == None: 
        prune(tree.trueBranch, minGain)

    if tree.falseBranch.results == None: 
        prune(tree.falseBranch, minGain)

    
    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        trueBranch, falseBranch = [], []

        # Pega os resultados dos nós folhas
        # v = resultado, c = quantidade daquele resultado
        for v, c in tree.trueBranch.results.items(): trueBranch += [[v]] * c
        for v, c in tree.falseBranch.results.items(): falseBranch += [[v]] * c

        p = float(len(trueBranch)) / len(trueBranch + falseBranch)

        # Calcula a diferença das taxas de erro
        delta = entropy(trueBranch + falseBranch) - p * entropy(trueBranch) - (1 - p) * entropy(falseBranch)
        
        # Realiza a poda da árvore caso a diferença das taxas seja menor que minGain
        if delta < minGain:
            # Desconsidera-se os nós filhos desse nó
            tree.trueBranch, tree.falseBranch = None, None
            tree.results = uniqueCounts(trueBranch + falseBranch)

def classify(observations, tree, dataMissing=False, accuracyTest=True):
    """
    Classifica as observações de acordo com a árvore de classificação gerada.
    
    Para accuracyTest == True, será retornado:
    - Um dicionário com informações de acurácia, número de acertos e erros.
    - Uma lista com os valores obtidos na classificação. 
    
    Para accuracyTest == False, será retornado:
    - Uma lista de classificações das observações dadas.
    
    Args:
        observation (list): observação a ser avaliada
        tree (DecisionTree): árvore de decisão gerada
        dataMissing (bool): True ou False se tiver dados faltando ou não
        accuracyTest (bool): True para realizar um teste de acurácia
    """

    def classifyWithoutMissingData(observations, tree):
        """ Classificação das observações que não possuem dados faltantes"""

        if tree.results != None:  # Nó folha que contém a classificação 
            return tree.results
        else:                     # Nó de decisão

            # Valor da observação correspondente ao atributo de divisão do nó 
            v = observations[tree.col]
            branch = None

            # Para valores do tipo int ou float
            if isinstance(v, int) or isinstance(v, float):
                """
                Se o valor da observação for maior ou igual ao valor de corte 
                daquele nó desce para a ramificação de valores verdadeiros.
                Se não, desce para a outra ramificação desse nó.
                """
                if v >= tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch

            # Para valores do tipo string
            else:
                """
                Se o valor da observação for igual ao valor de corte daquele nó
                desce para a ramificação de valores verdadeiros. 
                Se não, desce para a outra ramificação desse nó.
                """
                if v == tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch

        # Desce recursivamente na árvore
        return classifyWithoutMissingData(observations, branch)


    def classifyWithMissingData(observations, tree):
        """  Classificação das observações que possuem dados faltantes. """

        if tree.results != None:    # Nó folha com o resultado
            return tree.results
        else:                       # Nó de decisão
    
            # Valor da observação correspondente ao atributo de divisão do nó 
            v = observations[tree.col]
            if v == None:
                
                # Passa o exemplo com valor desconhecido para as ramificações do nó atual
                tr = classifyWithMissingData(observations, tree.trueBranch)
                fr = classifyWithMissingData(observations, tree.falseBranch)
                
                tcount = sum(tr.values())  
                fcount = sum(fr.values()) 
                
                # Cálculo da probabilidade com base na frequência observada dos valores para o atributo
                tw = float(tcount)/(tcount + fcount)
                fw = float(fcount)/(tcount + fcount)
                
                # defaultdict permite que seja criado um dicionario com valores padrões para as chaves
                result = defaultdict(int) 
                
                # Cálculo do 'voto' para cada classe
                for k, v in tr.items(): result[k] += v * tw
                for k, v in fr.items(): result[k] += v * fw
            
            else: # Realiza o mesmo processo de classifyWithoutMissingData()
                branch = None
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value: branch = tree.trueBranch
                    else: branch = tree.falseBranch
                else:
                    if v == tree.value: branch = tree.trueBranch
                    else: branch = tree.falseBranch

            # Desce recursivamente na árvore
            return classifyWithMissingData(observations, branch)


    """
    Seleciona a função de classificação de acordo com dataMissing

    As funções de classifyWithMissingData e classifyWithoutMissingData, 
    retornam um dicionário com a classificação daquele nó mais a quantidade
    daquele resultado do nó. Se um nó tiver mais de uma classificação, será 
    considerado o resultado com a maior quantidade.

    - Se for realizado um teste de acurácia o resultado obtido da classificação
    será comparado com o valor do resultado previamente dado. Um dicionário 
    será retornado com a quantidade de acertos, erros, e a acurária.
    - Se for realizado uma obtenção de classificação, retorna-se uma lista com
    todas as classificações das observações dadas.
    """
    
    def accuracy (observations, classifyData):
        """Realiza um teste de acurácia sobre os dados de teste."""
        predict = []
        
        trueClassify = 0
        falseClassify = 0
        
        for row in observations:
            
            # Obtém-se o valor da classificação
            value = classifyData(row[0:-1], tree) 

            # Caso aquele nó seja impuro, possui mais de uma classificação
            if len(value) > 1:
                if value[0.0] > value[1.0]:
                    if value[0.0] - value[1.0] > MARGEM_DE_ERRO:
                        result = 0
                    else:
                        result = 1
            else:
                result = int(list(value)[0])

            # Adiciona o resultado obtido a uma lista de resultados
            predict.append(float(result))
            
            # Incremento de classificações verdadeiras ou falsas
            if result == row[-1]: trueClassify += 1
            else: falseClassify += 1
                
                
        output = {
                    'Acertos': trueClassify, 
                    'Erros': falseClassify, 
                    'Acurácia': float(round(float(trueClassify)/len(observations), 4))
                }
                
        return output, predict
                
    def prediction (observations, classifyData):
        """Realiza a predição de dados sem conhecimento de seus resultados."""
        
        predict = []
        n = 1
        for row in observations:
            
            # Obtém-se o valor de classificação
            value = classifyData(row, tree)

            # Caso aquele nó possui mais de uma classificação
            if len(value) > 1:
                if value[0.0] > value[1.0]:
                    if value[0.0] - value[1.0] > MARGEM_DE_ERRO:
                        r = 0
                    else:
                        r = 1
            else:
                r = int(list(value)[0])

            result = 'Aluno com desempenho satisfatório' if r else 'Aluno com desempenho não satisfatório'
            
            # Adiciona o resultado obtido a uma lista de resultados
            predict.append(str(n) + 'ª Classificação: ' + result)
            n += 1
            
        return predict
        

    if dataMissing: # Se tiver dados faltantes no conjunto de observação
        if not accuracyTest: # Se for para obter a classificação de um cojunto
            return prediction(observations, classifyWithMissingData)

        else:
            return accuracy(observations, classifyWithMissingData)

    else: # Se não tiver dados faltantes no conjunto de observação
        if not accuracyTest: # Se for para obter a classificação de um cojunto
            return prediction(observations, classifyWithoutMissingData)

        else: # Se for um teste de acurácia
            return accuracy(observations, classifyWithoutMissingData)
            
import datetime

if __name__ == '__main__':

    dataCSV = pd.read_csv('dataset.csv')
    driver = Driver()
    driver.connect()
    start = int(datetime.datetime.now().timestamp())
    while True:

        now = int(datetime.datetime.now().timestamp())
        if now - start > 60 * 5:
            break

        # Separando dados de treinamento e de teste
        data = dataCSV.iloc[:, 0:-1].values
        results = dataCSV.iloc[:, -1].values
        trainingData, testData, trainingResults, testResults = train_test_split(data, results, test_size = 0.3)

        training = list(np.c_[trainingData, trainingResults])  # Concatena os dados com seus resultados
        evaluation = list(np.c_[testData, testResults])        # Concatena os dados com seus resultados

        # Geração da árvore de decisão    
        decisionTree = growDecisionTree(training)

        accuracy, predict = classify(evaluation, decisionTree, accuracyTest=True)
        cm = confusion_matrix(list(np.array(evaluation)[...,-1]), predict)

        params = {
            'id' : float(datetime.datetime.now().timestamp()),
            'name' : 'new_trees-02-07-2019',
            '0,0' : float(cm[0][0]),
            '0,1' : float(cm[0][1]),
            '1,0' : float(cm[1][0]),
            '1,1' : float(cm[1][1]),
        }
        driver.insert(params)