from scipy import *
from scipy.linalg import norm, pinv
from six.moves import xrange
import numpy as np
import timeit
import time
import random
from matplotlib import pyplot as plt

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [np.random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.centers_inited = False
        self.beta = 8
        self.W = np.random.random((self.numCenters, self.outdim))

    def refresh(self):
        self.W = np.random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        if len(d) != self.indim:
            print("input error {}/{}".format(len(d), self.indim))
            exit()
        return np.exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def setW(self, w):
        self.W = w.copy()

    def _makeCenters(self, X):
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]
        self.centers_inited = True

    def train(self, X, Y):
        if not self.centers_inited:
            print("初始化中心点")
            self._makeCenters(X)
        G = self._calcAct(X)
        self.W = np.dot(pinv(G), Y)

    def getG(self, X):
        return self._calcAct(X)

    def test(self, X):
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y

def zgsRBF(rbf, sx, sy, sw, times = 100, r = 0.01):
    size = len(sx)
    Graph = [[0,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
            [1,0,1,0]]
    HL = []
    for i in range(size):
        HL.append(np.matrix(rbf.getG(sx[i])))

    tmp = []
    for i in range(times):
        nw = [sw[n].copy() for n in range(len(sw))]
        for j in range(size):
            rbf.refresh()
            rbf.setW(nw[j])
            s = None
            for k in range(size):
                if j == k: continue
                if Graph[j][k] != 0 or Graph[k][j] != 0:
                    if s is None:
                        s = nw[k] - nw[j]
                    else:
                        s = s + nw[k] - nw[j]
            sw[j] = (nw[j] + r * (np.linalg.inv((np.dot(HL[j].T,HL[j])))) * s).A
    return sw

def main():
    left = 0
    right = 5
    n = 1000
    gap = (right - left) * 1.0 / n
    x = [i * gap for i in range(n)]
    random.shuffle(x)
    x = np.array(x).reshape(n,1)
    y = np.sin(x)
    '''
    # 直接求解
    start = time.process_time()
    rbf = RBF(1, 100, 1)
    rbf.train(x, y)
    end = time.process_time()
    print('[%d] Running time: %s Seconds' % (n, end - start))


    gap = (right - left) * 1.0 / n
    X = [gap * i for i in range(n)]
    Y = [np.sin(x) for x in X]
    t = rbf.test(np.array(X).reshape(n, 1))
    pY = [y[0] for y in t]

    plt.plot(X, Y)
    plt.plot(X, pY)
    plt.show()
    '''
    sx = np.array_split(x, 4)
    sy = []
    for t in sx:
        sy.append(np.sin(t))
    lw = []
    rbf = RBF(1, 30, 1)
    rbf.train(x,y)
    #w=rbf.W
    for i in range(len(sx)):
        tx = sx[i]
        print(len(tx))
        ty = sy[i]
        rbf.refresh()
        rbf.train(tx,ty)
        lw.append(rbf.W.copy())

    r = zgsRBF(rbf, sx, sy, lw, 500, 1e-30)

    '''
    # 直接求解
    start = time.process_time()
    rbf = RBF(1, 100, 1)
    rbf.train(x, y)
    end = time.process_time()
    print('[%d] Running time: %s Seconds' % (n, end - start))
    '''

    #rbf.setW(lw[1])
    rbf.setW(r[0])
    gap = (right - left) * 1.0 / n
    X = [gap * i for i in range(n)]
    Y = [np.sin(x) for x in X]
    t = rbf.test(np.array(X).reshape(n, 1))
    pY = [y[0] for y in t]
    print(pY)
    plt.plot(X, Y)
    plt.plot(X, pY)
    plt.show()


if __name__ == "__main__":
    main()
