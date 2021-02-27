import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

A = [[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]

def ShowNet(Arr):
    G = nx.from_numpy_matrix(np.array(Arr), create_using=nx.MultiDiGraph())
    pos = nx.circular_layout(G)
    nx.draw_circular(G,node_color=None)
    labels = {i: i + 1 for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=15)
    plt.show()

def ZGS(Graph, X, genTimes = 100, gama = 0.1):
    size = len(X)
    gsize = len(Graph)
    if gsize != size:
        print("图的大小和节点不匹配{},{}".format(size, gsize))
        exit(1)
    value_list = [X.copy()]
    for i in range(genTimes):
        flag = False
        nx = X.copy()
        for j in range(size):
            sum = 0
            for k in range(size):
                if j==k: continue
                if (Graph[j][k] != 0):
                    sum += (nx[k] - nx[j])
            X[j] = 1 / 3 * (gama * sum + 3 * nx[j])
        value_list.append(X.copy())
    return value_list

def Show(values, s, f1 = True, f2 = True):
    times = len(values)
    size = len(values[0])
    index = [i + 1 for i in range(times)]

    errs = []
    for i in range(size):
        tmp = []
        for j in range(times):
            e = (values[j][i] - s) ** 2
            tmp.append(e)
        errs.append(tmp)
    if f1:
        for i in range(size):
            plt.plot(index, errs[i],label='$(x_{} - x^*)^2$'.format("{{{}}}".format(i+1)))
        plt.legend(loc='upper right')
        plt.xlabel('iterations times')
        plt.ylabel('Error')
        plt.show()

    err = []
    for i in range(times):
        tmp = 0
        for j in range(size):
            tmp += (values[i][j] - s) ** 2
        err.append(tmp)
    if f2:
        plt.plot(index, err, label='$sum_i (x_i - x^*)^2$')
        plt.show()
    return errs, err

def Test(G, times = 300):
    Y = [8.7, 4.5, 2.8, 0.2, 2.8, 6.1, 2.9, 0.6]
    sum = 0
    for a in Y:
        sum += a
    sum = sum / (3.0 * len(Y))
    print(sum)
    for i in range(len(Y)):
        Y[i] = Y[i]/3 + 0.0001*random.random()
    r = ZGS(G, Y , times)
    es, e = Show(r, sum, False, False)
    return e

def CountSide(G):
    size = len(G)
    sum = 0
    for i in range(size):
        for j in range(size):
            if G[i][j] != 0:
                sum += 1
    return sum

if __name__ == "__main__":
    GS = []

    GS.append([[0,1,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0],
         [0,0,0,1,0,0,0,0],
         [0,0,0,0,1,0,0,0],
         [0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,1],
         [1,0,0,0,0,0,0,0]])

    GS.append([[0,1,0,0,1,0,0,0],
         [0,0,1,0,0,0,0,0],
         [0,0,0,1,0,0,0,0],
         [0,0,0,0,1,0,0,0],
         [1,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,1],
         [1,0,0,0,0,0,0,0]])

    GS.append([[0,1,1,0,0,0,0,0],
         [0,0,1,0,0,0,0,0],
         [0,0,0,1,1,0,0,0],
         [0,0,0,0,1,0,0,0],
         [1,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,1],
         [1,0,0,0,0,0,0,0]])

    GS.append([[0,1,1,0,0,0,0,0],
         [0,0,1,1,0,0,0,0],
         [0,0,0,1,1,0,0,0],
         [0,0,0,0,1,0,1,0],
         [1,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1,0],
         [0,1,0,0,0,0,0,1],
         [1,0,0,0,0,0,0,0]])

    GS.append([[0,1,1,0,0,0,0,0],
         [1,0,1,1,0,0,0,0],
         [0,0,0,1,1,0,0,0],
         [0,0,0,0,1,0,1,0],
         [1,0,0,0,0,1,0,0],
         [0,1,0,0,0,0,1,0],
         [0,1,0,0,0,0,0,1],
         [1,0,0,0,0,1,0,0]])

    times = 150
    index = [i for i in range(times + 1)]
    for i in range(len(GS)):
        e = Test(GS[i], times)
        plt.plot(index[:40], e[:40] , label='side = {}'.format(CountSide(GS[i])))

    plt.legend(loc='upper right')
    plt.xlabel('iterations times')
    plt.ylabel('Error')
    plt.show()
