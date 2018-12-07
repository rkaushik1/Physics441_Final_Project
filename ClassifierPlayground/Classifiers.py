from abc import ABC, abstractmethod

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

        self.tc = tcs[np.where(np.diff(np.sign(fa - fb)) != 0)[0][0]]

        return self

    def _t(self, data):
        return np.dot(self.a, data.transpose())

    def classify(self, x):
        return (self._t(np.array(x)) > self.tc) * 1 + 1

    def visualize(self):
        return plt.plot(self.means[:,0], self.means[:,1], 'r->')
