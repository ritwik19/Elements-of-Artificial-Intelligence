# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [Ritwik Budhiraja] -- [rbudhira]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    return np.linalg.norm(x1 - x2)


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    return np.sum(np.absolute(x1-x2))


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    x = np.clip(x, -1e100, 1e100)
    if derivative:
        return np.ones(x.shape)
    else:
        return x


#reference: https://www.v7labs.com/blog/neural-networks-activation-functions
def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    x = np.clip(x, -1e100, 1e100)
    sig = 1 / (1 + np.exp(-x))
    if derivative:
        return sig * (1 - sig)
    return sig


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    x = np.clip(x, -1e100, 1e100)
    tan_h = np.tanh(x)
    if derivative:
        return 1 - tan_h ** 2
    return tan_h


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    x = np.clip(x, -1e100, 1e100)
    r = np.maximum(x, 0)
    if derivative:
        return np.greater(x, 0).astype(int)
    return r


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


#reference: https://deepnotes.io/softmax-crossentropy#cross-entropy-loss
def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    p = np.clip(p, 1e-5, 1 - 1e-5)
    cross_entropy = -np.sum(y * np.log(p)) / p.shape[0]
    return cross_entropy


#reference: https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python
def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """

    y_one_hot = np.zeros((y.shape[0],y.max()+1))
    y_one_hot[np.arange(y.shape[0]),y] = 1
    return y_one_hot

