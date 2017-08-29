# encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from matplotlib.colors import ListedColormap

# load Iris dataset directly from the UCI Machine Learning Repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)
df.tail()

# extract the first 100 class label
y = df.iloc[0:150, 4].values
# convert the class labels [1 (Versicolor) and -1 (Setosa)]
y = np.where(y =='Iris-setosa', -1, 1)
# extract the first feature column (sepal length) and the third feature column (petal length)
x = df.iloc[0:150, [0, 2]].values

# visualize the matrix x in a two-dimensional scatter plot
plt.scatter(x[:50,0], x[:50,1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:150,0], x[50:150,1], color='blue', marker='x', label='Nao-setosa')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


# train our perceptron algorithm on the Iris data
ppn = Perceptron(0.1, 10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn._errors) + 1), ppn._errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()


# function to visualize the decision boundaries for 2D datasets
def plot_decision_regions(x, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors  = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])



    # determine the minimum and maximum values for the two features
    x1_min, x1_max = x[:,0].min() -1, x[:,0].max() + 1
    x2_min, x2_max = x[:,1].min() -1, x[:,1].max() + 1
    # use those feature vectors to create a pair of grid arrays xx1 and xx2
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    # plot the decision surface
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker = markers[idx], label=cl)


plot_decision_regions(x,y,classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()






