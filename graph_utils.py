__author__ = "shekkizh"

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def create_distance_matrix(X, Y=None, p=2):
    """
    Create distance matrix
    :param X:
    :param Y:
    :param metric:
    :return:
    """
    if Y is None:
        Y = X
    W = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i, Y.shape[0]):
            W[i, j] = lp_distance(X[i, :], Y[j, :], p)
    W = W + W.T
    return W


def create_directed_KNN_mask(D, knn_param=10, D_type='distance'):
    if D_type == 'similarity':
        directed_KNN_mask = np.argpartition(-D, knn_param + 1, axis=1)[:, 0:knn_param + 1]
    else:
        directed_KNN_mask = np.argpartition(D, knn_param + 1, axis=1)[:, 0:knn_param + 1]
    return directed_KNN_mask


def lp_distance(pointA, pointB, p):
    """
    Function to calculate the lp distance between two points
    :param p: the norm type to  calculate
    :return: distance
    """
    dist = (np.sum(np.abs(pointA - pointB) ** p)) ** (1.0 / p)

    return dist