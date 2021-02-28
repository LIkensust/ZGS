from scipy import *
from scipy.linalg import norm, pinv
from six.moves import xrange
import numpy as np
import timeit
import time
import random
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

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

    def makeCenters(self, X):
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]
        self.centers_inited = True

    def train(self, X, Y):
        if not self.centers_inited:
            print("初始化中心点")
            self.makeCenters(X)
        G = self._calcAct(X)
        self.W = np.dot(pinv(G), Y)

    def getG(self, X):
        return self._calcAct(X)

    def test(self, X):
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y

def zgsRBF(rbf, sx, sy, times = 100, r = 0.01):
    size = len(sx)
    QL = []
    sw = []
    rw = []
    for i in range(size):
        H = np.matrix(rbf.getG(sx[i]))
        Q = H.T * H + 0.01 * np.eye(H.shape[1])
        Q = Q.I
        sw.append((Q*H.T*sy[i]).A)
        rw.append((Q*H.T*sy[i]).A)
        QL.append(Q)

    tmp = [[] for i in range(size)]
    for i in range(times):
        nw = [sw[n].copy() for n in range(len(sw))]
        for j in range(size):
            s = np.zeros(nw[j].shape)
            for k in range(size):
                s = s + 0.5*(nw[k] - nw[j])
            sw[j] = (nw[j] + r * QL[j] * s).A
            tmp[j].append(sw[j][0][0])
    index = [i for i in range(len(tmp[0]))]
    #for i in range(size):
    #    plt.plot(index, tmp[i], label='{}'.format(i))
    #plt.legend(loc='upper right')
    #plt.show()
    #print(tmp[0][0],tmp[1][0],tmp[2][0],tmp[3][0])
    #print(tmp)
    return sw, rw

def main():
    left = -5
    right = 5
    n = 5000
    gap = (right - left) * 1.0 / n
    x = [i * gap for i in range(n)]
    random.shuffle(x)
    x = np.array(x).reshape(n,1)
    y = np.sin(x)*x
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
    sx = np.array_split(x, 5)
    sy = []
    for t in sx:
        sy.append(np.sin(t)*t)
    rbf = RBF(1, 100, 1)

    start = time.process_time()
    rbf.makeCenters(x)
    sw, rw = zgsRBF(rbf, sx, sy, 500, 0.005)
    end = time.process_time()
    print('分布式求解 [%d] Running time: %s Seconds' % (n, end - start))

    # 直接求解
    start = time.process_time()
    rbf.train(x, y)
    dw = rbf.W
    end = time.process_time()
    print('直接求解 [%d] Running time: %s Seconds' % (n, end - start))


    gap = (right - left) * 1.0 / n
    X = [gap * i for i in range(n)]
    Y = [np.sin(x)*x for x in X]

    fig, ax = plt.subplots(1, 1)


    aw = sw[0]
    for w in sw[1:]:
        aw  += w
    aw = aw / len(sw)
    rbf.setW(aw)
    t = rbf.test(np.array(X).reshape(n, 1))
    pY1 = [y[0] for y in t]
    ax.plot(X, pY1, label='Prediction after ZGS')
    rbf.setW(dw)
    t = rbf.test(np.array(X).reshape(n, 1))
    pY2 = [y[0] for y in t]
    ax.plot(X, pY2, label='Prediction for RBFN')
    '''
    for i in range(len(rw)):
        rbf.setW(rw[i])
        t = rbf.test(np.array(X).reshape(n, 1))
        pY = [y[0] for y in t]
        plt.plot(X, pY, '.' ,label='Prediction before ZGS {}'.format(i))
    '''
    ax.plot(X, Y,'--', label='Real image of the function')
    ax.legend(loc='upper right')
    ax.set_xlabel('x')
    ax.set_ylabel('xsin(x)')

    axins = ax.inset_axes((0.1, 0.7, 0.2, 0.2))
    zone_left = int(n / 10 * 1.5)
    zone_right = int(n / 10 * 2.5)
    print(zone_left,zone_right)
    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05  # y轴显示范围的扩展比例
    xlim0 = X[zone_left] - (X[zone_right] - X[zone_left]) * x_ratio
    xlim1 = X[zone_right] + (X[zone_right] - X[zone_left]) * x_ratio
    y = pY1[zone_left:zone_right]
    axins.plot(X[zone_left:zone_right],pY1[zone_left:zone_right])
    axins.plot(X[zone_left:zone_right],pY2[zone_left:zone_right])
    axins.plot(X[zone_left:zone_right],Y[zone_left:zone_right],'--')
    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    xy = (xlim1, ylim0)
    xy2 = (xlim1, ylim0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)



    axins = ax.inset_axes((0.65, 0.1, 0.2, 0.2))
    zone_left = int(n / 10 * 9.5)
    zone_right = int(n / 10 * 9.99)
    print(zone_left,zone_right)
    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05  # y轴显示范围的扩展比例
    xlim0 = X[zone_left] - (X[zone_right] - X[zone_left]) * x_ratio
    xlim1 = X[zone_right] + (X[zone_right] - X[zone_left]) * x_ratio
    y = pY1[zone_left:zone_right]
    axins.plot(X[zone_left:zone_right],pY1[zone_left:zone_right])
    axins.plot(X[zone_left:zone_right],pY2[zone_left:zone_right])
    axins.plot(X[zone_left:zone_right],Y[zone_left:zone_right],'--')
    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (xlim0, ylim0)
    xy2 = (xlim1, ylim0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    xy = (xlim0, ylim1)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    plt.show()


if __name__ == "__main__":
    main()
