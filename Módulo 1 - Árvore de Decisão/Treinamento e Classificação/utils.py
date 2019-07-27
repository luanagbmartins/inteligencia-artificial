# Código base foi escrito por Michael Dorner
# https://github.com/michaeldorner/DecisionTrees
#
# A função dotgraph é uma modificação realizada por Stanley Luck 
# https://github.com/lucksd356/DecisionTrees
#

import csv
import pydotplus
from collections import defaultdict
import pandas as pd

def loadCSV(file):
    """
    Carrega um arquivo CSV 
    """
    
    def convertTypes(s):
        """
        convertType() converte uma string em dados do tipo int ou float
        """
        
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

def dotgraph(decisionTree, headings):
    """
    Plota a árvore de decisão gerada
    
    Args:
        decisionTree (DecisionTree): árvore de decisão binária
    """

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

