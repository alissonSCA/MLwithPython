from sklearn import datasets
import numpy as np
from Perceptron import Perceptron

iris = datasets.load_iris()
X = iris.data
y = np.where(iris.target == 1, 1, -1)
print y

slp = Perceptron()
slp.fit(X,y)
teste = 10
errors = 0
for xi, yi in zip(X,y):
    yh = slp.predict(xi)
    errors += int(yi != yh)
print errors