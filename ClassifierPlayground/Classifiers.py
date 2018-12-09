from abc import ABC, abstractmethod

from sklearn import tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Classifier(ABC):
    @abstractmethod
    def train(self, data1, data2):
        pass

    @abstractmethod
    def classify(self, x):
        pass

    @abstractmethod
    def visualize(self):
        pass

class Fischer(Classifier):
    def train(self, data1, data2):
        self.data1 = pd.DataFrame(data1, columns=['x', 'y'])
        self.data2 = pd.DataFrame(data2, columns=['x', 'y'])

        within = self.data1.cov() + self.data2.cov()
        self.means = np.array([self.data2.mean(), self.data1.mean()])
        self.a = np.dot(np.linalg.inv(within.values), 
                        self.data2.mean() - self.data1.mean())

        ts = {1: self._t(self.data1.values),
              2: self._t(self.data2.values)}

        tcs = np.linspace(np.min(list(ts.values())), 
                          np.max(list(ts.values())), 10)
        fa = np.array([sum(ts[1] > tc) for tc in tcs])
        fb = np.array([sum(ts[2] < tc) for tc in tcs])

        self.tc = tcs[np.argmin(fa +fb)]

        return self

    def _t(self, data):
        return np.dot(self.a, data.transpose())

    def classify(self, x):
        return (self._t(np.array(x)) > self.tc) * 1 + 1

    def visualize(self):
        x = np.linspace(min(np.append(self.data1.x, self.data2.x)),
                        max(np.append(self.data1.x, self.data2.x)), 100 )
        y = (self.tc - x * self.a[0]) / self.a[1]
        return plt.plot(x, y, label='Decision Boundary', color='pink')

class Tree(Classifier):
    def train(self, data1, data2):
        self.data1 = pd.DataFrame(data1, columns=['x', 'y'])
        self.data2 = pd.DataFrame(data2, columns=['x', 'y'])

        self.clf = tree.DecisionTreeClassifier(max_depth=4)
        self.clf.fit(np.concatenate((self.data1.values, self.data2.values)), 
                     np.concatenate((np.ones(len(self.data1)), np.ones(len(self.data2.values)) * -1)))

        return self

    def classify(self, x):
        return self.clf.predict(x)

    def visualize(self):
        return
