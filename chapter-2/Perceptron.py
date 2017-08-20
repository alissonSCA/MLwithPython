# encoding: utf-8
import numpy as np

#Classificador Perceptron
class Perceptron(object):

    #Hiperparâmetros
    alpha = 0.0 #taxa de aprendizagem (entre 0 e 1)
    nIter = 0   #máximo de iterações

    #Parâmetros
    _w      = [] #vetor de pesos
    _errors = [] #número de erros de classificação em cada época de treinamento

    #Métodos

    #Construtor
    def __init__(self, alpha = 0.01, nIter = 10):
        self.alpha = alpha
        self.nIter = nIter

    #Treinamento
    def fit(self, X, y):
        #X: Matriz com nSamples linhas e nFeatures colunas
        #y: Vetor com tamanho nSample contendo +1 ou -1 de acordo com a classe de cada amostra
        self._w = np.zeros(1 + X.shape[1]) #Inicializa w com zeros
        self._errors = [] #Inicializa vetor de erros por época
        for _ in range(self.nIter):
            errors = 0 #Inicializa o erro desta época com zero
            for xi, yi in zip(X,y): #Comando zip forma pares com os elementos de X e de y
                yh = self.predict(xi)
                e = yi - yh
                self._w[1:]  += self.alpha*e*xi
                self._w[0]   += self.alpha*e*1
                errors += int(yi != yh)
            self._errors.append(errors)
        return self
    #Calcula o z
    def _sum2z(self,X):
        return np.dot(X, self._w[1:]) + self._w[0]
    #Teste
    def predict(self, X):
        return np.where(self._sum2z(X) >= 0, 1, -1)
