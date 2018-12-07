from abc import ABC, abstractmethod
from scipy import stats

import numpy as np

class DistributionGenerator(ABC):
    @abstractmethod
    def sample(self, n):
        pass


class TwoGaussGenerator(DistributionGenerator):
    def sample(self, n):
        return self._dist1(100), self._dist2(100, 6)

    def _dist1(self, n, mu=0):
        return np.random.multivariate_normal([mu, mu], np.array([[1,1],[1,2.5]]), n)

    def _dist2(self, n, mu=3):
        return np.random.multivariate_normal([mu**(.25), mu], np.array([[.5,-1],[-1,4]]), n)
