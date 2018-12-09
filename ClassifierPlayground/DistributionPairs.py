from abc import ABC, abstractmethod
from scipy import stats

import numpy as np

class DistributionGenerator(ABC):
    @abstractmethod
    def sample(self, n):
        pass
    

class TwoGaussGenerator(DistributionGenerator):
    def sample(self, n):
        return self._dist1(n), self._dist2(n, 6)

    def _dist1(self, n, mu=0):
        return np.random.multivariate_normal([mu, mu], np.array([[1,1],[1,2.5]]), n)

    def _dist2(self, n, mu=6):
        return np.random.multivariate_normal([mu**(.25), mu], np.array([[.5,-1],[-1,4]]), n)

class SeparatedGenerator(DistributionGenerator):
    def _dist1(self, n):
        return np.random.multivariate_normal([0, 0], np.array([[1,0],[0,1.5]]), n)

    def _dist2(self, n):
        n1 = n2 = n//2
        if n1 + n2 < n:
            n1 += 1

        return np.concatenate((np.random.multivariate_normal([-6, 0], np.array([[1,0],[0,1.75]]), n1),
                               np.random.multivariate_normal([6, 0], np.array([[1,0],[0,1.75]]), n2)))

    def sample(self, n):
        return self._dist1(n), self._dist2(n)

class CircularGenerator(DistributionGenerator):
    def _cartesian(self, r, theta):
        return r * np.cos(theta), r * np.sin(theta)

    def _dist1(self, n):
        return np.random.multivariate_normal([0, 0], np.array([[1,0],[0,1]]), n)

    def _dist2(self, n):
        ts = np.random.rand(n) * np.pi * 2
        rs = np.sqrt(np.abs(stats.norm.rvs(scale=9, size=n) + 25))
        return np.dstack((self._cartesian(rs, ts)))[0]

    def sample(self, n):
        return self._dist1(n), self._dist2(n)
