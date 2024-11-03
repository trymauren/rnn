import numpy as np
from scipy.special import expit, softmax


class Activation():
    """ Wrapper class for activation functions """

    def __init__(self):
        super().__init__()

    def __call__(self, z):
        return self.eval(z)


class Relu(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """ Implements g(x) = max(0, z) """
        return np.maximum(0, z)

    def grad(self, a: list):
        """
        Implements g'(z) = 1 : x >= 0, g'(z) = 0 : x < 0
        Decided on using g'(0) = 1. Could do g'(0) = 0 instead
        """
        return np.array([1 if i >= 0 else 0 for i in a[0]])

class Tanh(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """ Implemens g(z) = tanh(z) """
        return np.tanh(z)

    def grad(self, a):
        """ Implements g'(z) = 1-z^2 """
        return 1 - np.square(a)


class Sigmoid(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """ Implements g(z) = 1(1+e^{-z}) """
        return expit(z)

    def grad(self, a):
        """ Implements g'(z) = z(1-z) """
        return a*(1 - a)


# ----------- "Activation" functions used on output layer ----------- #

class Identity(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """ Implements g(z) = z """
        return z


class Softmax(Activation):

    def __init__(self):
        super().__init__()

    def eval(self, z):
        """ Implements g(z) = softmax(z) """
        return softmax(z, axis=-1)
