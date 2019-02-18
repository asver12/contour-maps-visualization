from scipy.stats import multivariate_normal

import numpy as np


def get_gaussian(x_min, x_max, y_min, y_max, mu_x=0.0, variance_x=0.0, mu_y=0.0, variance_y=0.0, size=500000):
    xlist = np.linspace(x_min, x_max, size)
    ylist = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(xlist, ylist)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    Z = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
    return X, Y, Z.pdf(pos)


def find_index(number, levels, verbose = False):
    start = 0
    end = len(levels)-1
    if number < levels[start]:
        return 0
    if number > levels[end]:
        return end + 1
    pivo = int((end - start)/2+start)
    while start != pivo:
        if verbose:
            print("{}:{}:{}".format(start,pivo,end))
        if number < levels[pivo]:
            end = pivo
        if number > levels[pivo]:
            start = pivo
        pivo = int((end - start)/2+start)
    return pivo + 1