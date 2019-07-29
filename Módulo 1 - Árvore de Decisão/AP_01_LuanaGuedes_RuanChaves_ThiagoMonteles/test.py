import csv 
import pandas as pd


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

import pandas as pd
import numpy as np 
import seaborn as sns
import math
from collections import defaultdict
import random
import pydotplus
from IPython.display import Image, display

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Permitir downsampling / oversampling ?
ALLOW_SAMPLING = False

# Parâmetros da função de sampling
MAX_DISTANCE = 0.5
OVERSAMPLING = False
OVERSAMPLING_LIMIT = 1.0

arquivo = random.choice(["dataset_reduzido.csv"])


def loadCSV(file):
    """ Carrega um arquivo CSV"""
    
    def convertTypes(s):
        """ Converte uma string em dados do tipo int ou float """
        
        s = s.strip() # Função strip retorna uma string sem espaços
        try:
            # se a string for do tipo x.y, onde x e y são números, converte-se a string para um número flutuante
            # se não houver a presença de um .(ponto), então converte-se em um número inteiro
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt')) # abre arquivo csv

    # next() itera sobre uma lista, no caso, o arquivo csv lido
    # lsHeader irá receber toda a primeira linha do arquivo csv que contém o nome dos atributos
    listHeader = next(reader) 
    header = {}

    # Cria-se um dicionario enumerando todos os atributos 
    # Ex: {'Column 0': 'Atributo0', 'Column 1': 'Atributo1', 'Column 2': 'Atributo2'}
    for it, attribute in enumerate(listHeader):
            column = 'Column %d' % it
            header[column] = str(attribute)

    # Para cada item de cada linha do arquivo csv, converte-se a string para o número associado
    dataset = [[convertTypes(item) for item in row] for row in reader]
    dataset = pd.DataFrame(data=dataset)

    return header, dataset


# Carrega arquivo csv e obtém um dicionario de atributos e os dados de treinamento
headings, dataCSV = loadCSV(arquivo)

from sklearn.model_selection import train_test_split
import numpy as np


# Separando dados de treinamento e de teste
data = dataCSV.iloc[:, 0:-1].values
results = dataCSV.iloc[:, -1].values
trainingData, testData, trainingResults, testResults = train_test_split(data, results, test_size = 0.2)

training = list(np.c_[trainingData, trainingResults])  # Concatena os dados com seus resultados
evaluation = list(np.c_[testData, testResults])        # Concatena os dados com seus resultados

def sampling(training, evaluation):
    training_vectors = [x[0:-1] for x in training]
    labels = [ x[-1] for x in training ]
    # D é uma matriz onde a posição (i,j) é a distância euclideana entre 
    # os vetores na posição i e j da matriz original.
    D = euclidean_distances(training_vectors)

    # Identificar os vetores que podem ser deletados:
    # E é uma matriz booleana: 1 caso o par (i,j) sejam dois elementos de classe distinta,
    # e valor 0 caso o par (i,j) contenha dois elementos de mesma classe.
    E = []
    for x in labels:
        row = []
        for y in labels:
            if int(x) ^ int(y): # operador XOR
                row.append(1)
            else:
                row.append(0)
        E.append(row)

    remove_list = []
    for i,j in enumerate(D):
        if training[i][-1] == 0:
            continue
        index = [x for x in range(len(j))]
        vector = list(zip(j,E[i],index))
        vector = [x for x in vector if x[1] == 1] # pegar vetores de classe distinta
        vector = sorted(vector, key=lambda x: x[0]) # ordenar por distância
        # capturar vetor distinto mais próximo que ainda não foi capturado
        for item in vector:
            if item[2] not in remove_list:
                if item[0] < MAX_DISTANCE: # não remover vetores muito distantes
                    remove_list.append(item[2])
                break

    assert(all([training[x][-1] == 0 for x in remove_list]))

    # Remoção dos vetores majoritários
    training = [v for i,v in enumerate(training) if i not in remove_list]
    
    minor_list = []
    if not OVERSAMPLING:
        evaluation = random.sample(evaluation, len(evaluation) - len(remove_list))
        return training, evaluation
    
    #Introdução dos vetores minoritários
    D_samples = [ (i,v) for i,v in enumerate(D)]
    lim = int(OVERSAMPLING_LIMIT * len(remove_list))
    for idx in range(lim):
        chosen_vector = random.choice(D_samples)
        i = chosen_vector[0]
        j = chosen_vector[1]
        if labels[i] == 0:
            continue
        index = [x for x in range(len(j))]
        vector = list(zip(j,E[i],index))
        vector = [x for x in vector if x[1] == 0] # pegar vetores de mesma classe
        vector = [x for x in vector if x[0] != 0.0] # não pegar a si mesmo
        vector = sorted(vector, key=lambda x: x[0]) # ordenar por distância
        # capturar vetor de mesma classe mais próximo que ainda não foi capturado
        # colocar o vetor médio entre os dois no conjunto de treinamento
        for item in vector:
            if item[2] not in minor_list:
                selected_vector = training_vectors[item[2]]
                a = np.array(selected_vector)
                b = np.array(training_vectors[i])
                c = list((a + b) / 2)
                c.append(1)
                training.append(c)
                minor_list.append(item[2])
                break    
    return training, random.sample(evaluation, len(evaluation) - lim)

if ALLOW_SAMPLING:
    training, evaluation = sampling(training, evaluation)

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


# Geração da árvore de decisão    
decisionTree = growDecisionTree(training) 



def dotgraph(decisionTree, headings):
    """ Plota a árvore de decisão gerada."""

    dcNodes = defaultdict(list)
    def toString(iSplit, decisionTree, bBranch, szParent = "null", indent=''):
        if decisionTree.results != None:  # leaf node
            lsX = [(x, y) for x, y in decisionTree.results.items()]
            lsX.sort()
            szY = ', '.join(['%s: %s' % (x, y) for x, y in lsX])
            dcY = {"name": szY, "parent" : szParent}
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append(['leaf', dcY['name'], szParent, bBranch, dcSummary['impurity'],
                                    dcSummary['samples']])
            return dcY
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in headings:
                    szCol = headings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                    decision = '%s >= %s' % (szCol, decisionTree.value)
            else:
                    decision = '%s == %s' % (szCol, decisionTree.value)
            trueBranch = toString(iSplit+1, decisionTree.trueBranch, True, decision, indent + '\t\t')
            falseBranch = toString(iSplit+1, decisionTree.falseBranch, False, decision, indent + '\t\t')
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append([iSplit+1, decision, szParent, bBranch, dcSummary['impurity'],
                                    dcSummary['samples']])
            return

    toString(0, decisionTree, None)
    
    lsDot = ['digraph Tree {',
                'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
                'edge [fontname=helvetica] ;'
    ]

    i_node = 0
    dcParent = {}
    for nSplit in range(len(dcNodes)):
        lsY = dcNodes[nSplit]
        for lsX in lsY:
            iSplit, decision, szParent, bBranch, szImpurity, szSamples =lsX
            if type(iSplit) == int:
                szSplit = '%d-%s' % (iSplit, decision)
                dcParent[szSplit] = i_node
                lsDot.append('%d [label=<%s<br/>impurity %s<br/>samples %s>, fillcolor="#e5813900"] ;' % (i_node,
                                        decision.replace('>=', '&ge;').replace('?', ''),
                                        szImpurity,
                                        szSamples))
            else:
                lsDot.append('%d [label=<impurity %s<br/>samples %s<br/>class %s>, fillcolor="#e5813900"] ;' % (i_node,
                                        szImpurity,
                                        szSamples,
                                        decision))
                
            if szParent != 'null':
                if bBranch:
                    szAngle = '45'
                    szHeadLabel = 'True'
                else:
                    szAngle = '-45'
                    szHeadLabel = 'False'
                szSplit = '%d-%s' % (nSplit, szParent)
                p_node = dcParent[szSplit]
                if nSplit == 1:
                    lsDot.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node,
                                                        i_node, szAngle, szHeadLabel))
                else:
                    lsDot.append('%d -> %d ;' % (p_node, i_node))
            i_node += 1
    lsDot.append('}')
    dot_data = '\n'.join(lsDot)
    return dot_data

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

# Poda da árvore (método de pós-poda) 
prune(decisionTree, 0.5)                       

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
            if len(value) > 1: result = 0.0 if value[0.0] > value[1.0] else 1.0
            else: result = list(value)[0]

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
            if len(value) > 1: r = 0 if value[0.0] > value[1.0] else 1
            else: r = int(list(value)[0])

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
            
# Realiza teste de acurácia com os dados de teste
accuracy, predict = classify(evaluation, decisionTree, accuracyTest=True)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(list(np.array(evaluation)[...,-1]), predict) 
ax = sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues')
ax.set(ylabel='Valor verdadeiro')
ax.set(xlabel='Valor previsto')
labels = ['Verdadeiro Negativo', 
          'Falso Positivo', 
          'Falso Negativo', 
          'Verdadeiro Positivo']
ax.set(yticklabels=['Negativo','Positivo'])
ax.set(xticklabels=['Negativo','Positivo'])
count = 0
bacc = []
for idx, text in enumerate(ax.texts):
    label = int(text.get_text())
    count += label
    bacc.append(label)

print(bacc)
acc1 = bacc[0] / ( bacc[0] + bacc[2] )
acc2 = bacc[3] / ( bacc[1] + bacc[3] ) 

import json
try:
    open('results.json','r')
except:
    with open('results.json', 'w+') as f:
        json.dump([], f)

with open('results.json','r') as f:
    l = json.load(f)

result = {
    "arquivo": arquivo,
    "Acurácia 1": acc1,
    "Acurácia 2": acc2,
    "Acurácia Total": accuracy['Acurácia'],
    "Acurácia Equilibrada": ( acc1 + acc2 ) / 2
}

l.append(result)
with open('results.json','w') as f:
    json.dump(l, f)