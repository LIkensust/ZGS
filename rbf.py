from scipy import *
from scipy.linalg import norm, pinv
from six.moves import xrange
import numpy
import timeit
import time

from matplotlib import pyplot as plt
class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        if len(d) != self.indim:
            print("input error {}/{}".format(len(d), self.indim))
            exit()
        return numpy.exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        G = numpy.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]
        G = self._calcAct(X)
        self.W = numpy.dot(pinv(G), Y)

    def test(self, X):
        G = self._calcAct(X)
        Y = numpy.dot(G, self.W)
        return Y


def Test(n):
    x = mgrid[0:3:complex(0, n)].reshape(n, 1)
    y = numpy.sin(x)
    start = time.process_time()
    rbf = RBF(1, 100, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    end = time.process_time()
    print('[%d] Running time: %s Seconds' % (n,end - start))

    X = [0.03 * i for i in range(100)]
    Y = [numpy.sin(x) for x in X]
    t = rbf.test(numpy.array(X).reshape(100, 1))
    pY = [y[0] for y in t]

    plt.plot(X,Y)
    plt.plot(X,pY)
    plt.show()

    return end-start

if __name__ == '__main__':
    Test(100)

