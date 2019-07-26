# Código base foi escrito por Michael Dorner
# https://github.com/michaeldorner/DecisionTrees
#
# Algumas modificações realizadas, como a função dotgraph, foram realizadas por
# Stanley Luck 
# https://github.com/lucksd356/DecisionTrees
#

from math import log


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


def divideSet(rows, column, value):
    """
    Realiza a separação dos atributos dado um valor de comparação.

    Args:
        rows (): coleção de dados
        column (int): quantidade de atributos
        value: valor a ser usado como comparação
    
    Returns:
        list: coleção de dados separados em dois

    """

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
    """
    Recebe uma quantidade de dados e calcula a quantidade de cada resultado ali 
    presente.

    Args:
        rows: coleção de dados
    
    Returns: 
        Dict: atributo de resposta com sua quantidade correspondente
    """

    results = {}
    for row in rows:
        # A resposta da classificação estará na última coluna 
        # Cada resposta diz se o aluno é um bom ou mal aluno
        r = row[-1]
        if r not in results: results[r] = 0
        results[r] += 1

    return results


def entropy(rows):
    """
    Cálculo do grau de entropia. O dado cálculo é definido por: 
    -somatorio(p(nó) * log2(p(nó)))

    Args:
        rows (): coleção de dados

    Returns:
        float: grau de entropia
    """

    log2 = lambda x: log(x) / log(2)        # Função para calcular log2(x)
    results = uniqueCounts(rows)            # Pega todos os resultados e sua respectiva quantidade

    entr = 0.0
    for r in results:
        p = float(results[r]) / len(rows)   # Probabilidade de um dado resultado 
        entr -= p * log2(p)                 # Calcula o grau da entropia dado por 

    return entr


def growDecisionTreeFrom(rows, evaluationFunction=entropy):
    """
    Gera uma árvore de decisão recursivamente realizando as divisões dos dados
    dado um atributo de divisão. 

    Algumas observações a serem consideradas:    
    - Como critério de parada, observa-se se não há ganho de informação 
    significativo, ou seja, se o ganho é maior que zero.
    - Para determinar o quão boa é uma condição de teste realizada, é preciso 
    comparar o grau de entropia do nó antes da divisão e dos nós gerados após a 
    divisão. O atributo que gerar uma maior diferença é escolhido como condição
    para teste. 

    Args:
        rows (): coleçao de dados
        evaluationFunction (function): função de avaliação

    Returns:
        DecisionTree: árvore de decisão
    """

    # Verifica se não é mais possível fazer o crescimento da árvore
    if len(rows) == 0: return DecisionTree()     

    # Realiza o cálculo do grau da entropia do nó pai
    currentScore = evaluationFunction(rows)     

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
            gain = currentScore - p*evaluationFunction(set1) - (1-p)*evaluationFunction(set2)

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
        trueBranch = growDecisionTreeFrom(bestSets[0], evaluationFunction)  

        # Ramificação com valores do tipo Falso
        falseBranch = growDecisionTreeFrom(bestSets[1], evaluationFunction)

        # Armazena o nó pai na árvore
        return DecisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch,
                            falseBranch=falseBranch, summary=summary)

    # Se não há ganho de informação, não há mais ramificação desse nó
    else:
        # Armazena o nó folha na árvore os resultados de classificação
        return DecisionTree(results=uniqueCounts(rows), summary=summary)


def prune(tree, minGain, evaluationFunction=entropy):
    """
    Realiza a poda da árvore de acordo com um valor mínimo estabelecido.
    A árvore é cortada, removendo-se as sub-ramificações que não contribuem 
    para a acurácia. O método utlizado chama-se 'cost-complexity pruning'.
    Para cada nó interno da árvore, calcula-se a taxa de erro caso a sub-árvore
    abaixo desse nó seja excluido. Depois calcula-se a taxa de erro caso não 
    haja poda. Se a diferença entre essas taxas for menor que o minGain, a 
    árvore é podada.

    Args:
        tree (DecisionTree): árvore de decisão
        minGain (float): entropia  
        evaluationFunction (function): função de avalição
    """

    # Olha se os filhos de um nó são folhas(resultado). Se None não é.
    if tree.trueBranch.results == None: 
        prune(tree.trueBranch, minGain, evaluationFunction)

    if tree.falseBranch.results == None: 
        prune(tree.falseBranch, minGain, evaluationFunction)

    
    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        trueBranch, falseBranch = [], []

        # Pega os resultados dos nós folhas
        # v = resultado, c = quantidade daquele resultado
        for v, c in tree.trueBranch.results.items(): trueBranch += [[v]] * c
        for v, c in tree.falseBranch.results.items(): falseBranch += [[v]] * c

        p = float(len(trueBranch)) / len(trueBranch + falseBranch)

        # Calcula a diferença das taxas de erro
        delta = evaluationFunction(trueBranch + falseBranch) - p * evaluationFunction(trueBranch) - (1 - p) * evaluationFunction(falseBranch)
        
        # Realiza a poda da árvore caso a diferença das taxas seja menor que minGain
        if delta < minGain:
            # Desconsidera-se os nós filhos desse nó
            tree.trueBranch, tree.falseBranch = None, None
            tree.results = uniqueCounts(trueBranch + falseBranch)


def classify(observations, tree, dataMissing=False):
    """
    Classifica as observações de acordo com a árvore.

    Args:
        observation: observação a ser avaliada
        tree (DecisionTree): árvore de decisão gerada
        dataMissing (bool): True ou False se tiver dados faltando ou não
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
        """ 
        Classificação das observações que possuem dados faltantes.

        Obs: o autor do código, Michael Dorner, utilizou a referência
        http://blog.ludovf.net/python-collections-defaultdict/
        porém o site encontra-se fora do ar.
        """

        if tree.results != None:    # Nó folha com o resultado
            return tree.results
        else:                       # Nó de decisão
    
            # Valor da observação correspondente ao atributo de divisão do nó 
            v = observations[tree.col]
            if v == None:
                tr = classifyWithMissingData(observations, tree.trueBranch)
                fr = classifyWithMissingData(observations, tree.falseBranch)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = float(tcount)/(tcount + fcount)
                fw = float(fcount)/(tcount + fcount)
                result = defaultdict(int) 
                for k, v in tr.items(): result[k] += v * tw
                for k, v in fr.items(): result[k] += v * fw

                return dict(result)
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

    # Seleciona a função de classificação de acordo com dataMissing
    if dataMissing:
        return classifyWithMissingData(observations, tree)
    else:
        return classifyWithoutMissingData(observations, tree)
