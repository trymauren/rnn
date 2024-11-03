import sys
import git
import numpy as np
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class Optimiser():

    def __init__(self):
        pass

    def __call__(self, params, **kwargs):
        """
        Calls the step() function of the implemented optimiser:
        Given a list 'params' of gradients for the parameters to be
        optimised, the step function returns a list of the same length
        with update values for the given parameters.

        """

        return self.step(params, **kwargs)


class SGD(Optimiser):

    def __init__(self):
        super().__init__()

    def step(self, params, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.update = [0]*len(params)
        for idx, param in enumerate(params):
            self.update[idx] = self.learning_rate*param
        return self.update


class SGD_momentum(Optimiser):

    def __init__(self):
        super().__init__()
        self.update = None

    def step(self, params, learning_rate=0.001, momentum_rate=0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        if self.update is None:
            self.update = [0]*len(params)

        momentum = [0]*len(params)
        for idx, param in enumerate(params):
            momentum = self.momentum_rate*self.update[idx]
            self.update[idx] = momentum+self.learning_rate*param
        return self.update


class AdaGrad(Optimiser):

    def __init__(self):
        super().__init__()
        self.delta_ = 1e-7
        self.alphas = None
        self.update = None

    def step(self, params, learning_rate=0.001):
        self.learning_rate = learning_rate
        if self.alphas is None:
            self.alphas = [0]*len(params)
            self.update = [0]*len(params)

        for idx, param in enumerate(params):
            self.alphas[idx] += np.square(param)
            adagrad = param / (self.delta_ + np.sqrt(self.alphas[idx]))
            self.update[idx] = self.learning_rate * adagrad
        return self.update


class RMSProp(Optimiser):

    def __init__(self):
        super().__init__()
        self.delta_ = 1e-6
        self.update = None
        self.alphas = None

    def step(self, params, learning_rate=0.001, decay_rate=0.001):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        if self.alphas is None:
            self.alphas = [0]*len(params)
            self.update = [0]*len(params)

        for idx, param in enumerate(params):
            self.alphas[idx] += (
                                 self.decay_rate * param
                                 + (1 - decay_rate)
                                 * np.square(param)
                                )
            rmsprop = param / (self.delta_ + np.sqrt(self.alphas[idx]))
            self.update[idx] = self.learning_rate * rmsprop
        return self.update


class Adam(Optimiser):
    def __init__(self):
        super().__init__()
        self.alphas1 = None      # s in Deep Learning book, algorithm 8.7
        self.alphas2 = None      # r in Deep Learning book, algorithm 8.7
        # self.update  = None
        self.t = 0

    def step(self,
             params,
             learning_rate=.001,
             decay_rate1=.9,
             decay_rate2=.999,
             delta=1e-8
             ):

        self.epsilon = learning_rate
        self.rho1 = decay_rate1
        self.rho2 = decay_rate2
        self.delta = delta

        if self.alphas1 is None:
            self.alphas1 = [0]*len(params)
        if self.alphas2 is None:
            self.alphas2 = [0]*len(params)

        self.update = []
        self.t = self.t + 1

        for idx, param in enumerate(params):
            self.alphas1[idx] = (
                self.rho1 * self.alphas1[idx]
                + (1 - self.rho1) * param
            )
            self.alphas2[idx] = (
                self.rho2 * self.alphas2[idx]
                + (1 - self.rho2) * np.square(param)
            )
            alpha1_hat = self.alphas1[idx]/(1 - self.rho1**self.t)
            alpha2_hat = self.alphas2[idx]/(1 - self.rho2**self.t)
            self.update.append(
                self.epsilon * (
                    alpha1_hat/(np.sqrt(alpha2_hat) + self.delta)
                )
            )
        return self.update


def clip_gradient(gradients: np.ndarray, threshold: float) -> np.ndarray:
    """
    Normalises (clips) the gradient argument given to 'gradients'.
    """
    for g in gradients:
        norm_g = np.linalg.norm(g)
        if norm_g > threshold:
            g *= threshold/norm_g
    return gradients