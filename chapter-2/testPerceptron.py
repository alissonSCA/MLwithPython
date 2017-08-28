# encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron

# load Iris dataset directly from the UCI Machine Learning Repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)
df.tail()

# extract the first 100 class label
y = df.iloc[0:100, 4].values
# convert the class labels [1 (Versicolor) and -1 (Setosa)]
y = np.where(y =='Iris-setosa', -1, 1)
# extract the first feature column (sepal length) and the third feature column (petal length)
x = df.iloc[0:100, [0, 2]].values

# visualize the matrix x in a two-dimensional scatter plot
plt.scatter(x[:50,0], x[:50,1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker='x', label='Versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

# train our perceptron algorithm on the Iris data subset
ppn = Perceptron(0.1, 10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn._errors) + 1), ppn._errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()





