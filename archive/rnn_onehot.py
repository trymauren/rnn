from importlib.resources import open_text
from pathlib import Path
import sys
import os
from typing import Dict
import git
import numpy as np
from collections.abc import Callable
import yaml
from utils.activations import Relu, Tanh, Identity, Softmax
from utils.loss_functions import Mean_Square_Loss as mse
from utils.loss_functions import Classification_Logloss as ce
from utils import optimisers
from utils.optimisers import SGD, SGD_momentum, AdaGrad
from utils import read_load_model
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
import matplotlib.pyplot as plt
from tqdm import tqdm

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
np.random.seed(24)


class RNN:

    def __init__(
            self,
            hidden_activation: Callable = None,
            output_activation: Callable = None,
            loss_function: Callable = None,
            optimiser: Callable = None,
            regression: bool = False,
            classification: bool = False,
            name: str = 'rnn',
            config: Dict | Path | str = 'default',
            seed: int = 24,
            threshold = 5,
            ) -> None:

        np.random.seed(seed)
        # Setting activation functions, loss function and optimiser
        if not hidden_activation:
            hidden_activation = Relu()
        self._hidden_activation = eval(hidden_activation)
        if not output_activation:
            output_activation = Tanh()
        self._output_activation = eval(output_activation)

        if not loss_function:
            loss_function = mse()
        self._loss_function = eval(loss_function)

        if not optimiser:
            optimiser = AdaGrad()
        self._optimiser = eval(optimiser)

        self.regression = regression
        self.classification = classification
        # Initialize weights and biases as None until properly
        # initialized in fit() method.
        # xh = input  -> hidden
        # hh = hidden -> hidden
        # hy = hidden -> output
        self.w_xh, self.w_hh, self.w_hy = None, None, None

        self.b_hh, self.b_hy = None, None

        self.xs, self.hs, self.ys = None, None, None

        self.built = False

        self.name = name

        self.threshold = threshold

        self.stats = {
            'other stuff': [],
        }

    def _forward(
            self,
            X_partition,
            generate=False,
            onehot=False,
            ) -> None:
        """
        Forward-pass method to be used in fit-method for training the
        RNN. Returns predicted output values

        Parameters:
        -------------------------------
        X_partition:
            - A partition of samples

        generate:
            - Whether to insert output at time t=1 as input at time t=2

        Returns:
        -------------------------------
        None
        """
        for t in range(self.num_hidden_states):
            x_weighted = self.w_xh @ X_partition[t]
            h_weighted = self.w_hh @ self.hs[t-1]
            z = x_weighted + h_weighted
            self.xs[t] = z
            h_t = self._hidden_activation(z)
            self.hs[t] = h_t
            self.ys[t] = self._output_activation(self.w_hy @ self.hs[t])

            if generate:
                if t < self.num_hidden_states - 1:
                    if onehot:
                        # ix = np.random.choice(range(len(self.ys[t])),
                        #                       p=self.ys[t].ravel())
                        ix = np.argmax(self.ys[t])
                        onehot_ys = np.zeros_like(self.ys[t])
                        onehot_ys[ix] = 1
                        X_partition[t+1] = onehot_ys
                        self.ys[t] = onehot_ys
                    else:
                        X_partition[t+1] = self.ys[t]
        return self.ys

    def _backward(self, num_backsteps=np.inf) -> None:

        deltas_w_xh = np.zeros_like(self.w_xh, dtype=float)
        deltas_w_hh = np.zeros_like(self.w_hh, dtype=float)
        deltas_w_hy = np.zeros_like(self.w_hy, dtype=float)

        deltas_b_hh = np.zeros_like(self.b_hh, dtype=float)
        deltas_b_hy = np.zeros_like(self.b_hy, dtype=float)

        prev_grad_h_Cost = np.zeros_like(self.num_hidden_nodes)

        #NOTE: Implemented gradient clipping, however shape error, 
        #      gradient norm is a list of floats, not one number
        loss_grad = self._loss_function.grad()
        num_backsteps = min(len(self.hs)-1, num_backsteps)

        for t in range(num_backsteps, -1, -1):

            """ BELOW IS CALCULATION OF GRADIENTS W/RESPECT TO HIDDEN_STATES
            - SEE (1-~20) IN TEX-DOCUMENT """

            """OUTDATED"""
            # """Just doing some copying. grad_o_Cost will, in the next
            # line of code, contain the cost vector"""
            # grad_o_Cost = np.copy(y_pred[t])

            # """See deep learning book, 10.18 for
            # explanation of following line. Also:
            # http://cs231n.github.io/neural-networks-case-study/#grad
            # Eventually, one can find grad(C) w/ respect to C^t"""
            # grad_o_Cost[y_true[t]] -= 1
            """OUTDATED END"""

            """ NEW """
            # grad_o_Cost = self._loss_function.grad()
            if self.regression:
                grad_o_Cost_t = loss_grad[:, t]
            if self.classification:
                print('not implemented error')
            """ NEW END """

            """A h_state's gradient update are both influenced by the
            preceding h_state at time t+1, as well as the output at
            time t. The cost/loss of the current output derivated with
            respect to hidden state t is what makes up the following
            line before the "+ sign". After "+" is the gradient through
            previous hidden states and their outputs. This term after
            the "+" sign, is 0 for first step of BPTT.

            Eq. 16 in tex-document(see also eq. 15 for first iteration of BPPT)
            Eq. 10.20 in DLB"""
            grad_h_Cost = prev_grad_h_Cost + self.w_hy.T @ grad_o_Cost_t
            grad_h_Cost = optimisers.clip_gradient(grad_h_Cost, self.threshold)
            # grad_h_Cost = optimisers.clip_gradient(grad_h_Cost, self.threshold)
            # print(prev_grad_h_Cost.shape)
            # print(self.w_hy.T.shape)
            # print(grad_h_Cost.shape)
            """The following line is to shorten equations. It fetches/
            differentiates the hidden activation function."""
            d_act = self._hidden_activation.grad(self.hs[t])

            """ BELOW IS CALCULATION OF GRADIENT W/RESPECT TO WEIGHTS """

            """Cumulate the error."""
            deltas_w_hy += self.hs[t].T * grad_o_Cost_t  # 10.24 in DLB
            deltas_w_hh += d_act @ self.hs[t-1] * grad_h_Cost  # 10.26 in DLB
            deltas_w_xh += d_act @ self.xs[t] * grad_h_Cost  # 10.28 in DLB
            deltas_b_hy += grad_o_Cost_t * 1  # 10.22 in DLB
            deltas_b_hh += d_act @ grad_h_Cost  # 10.22 in DLB

            """Pass on the bits of the chain rule to the calculation of
            the previous hidden state update
            This line equals the first part of eq. 10.21 in DLB
            To emphasize: the part before the "+" in 10.21 in DLB"""
            prev_grad_h_Cost = d_act @ self.w_hh.T @ grad_h_Cost
            prev_grad_h_Cost = prev_grad_h_Cost

        params = [self.w_hy, self.w_hh, self.w_xh,
                  self.b_hy, self.b_hh]
        deltas = [deltas_w_hy, deltas_w_hh, deltas_w_xh,
                  deltas_b_hy, deltas_b_hh]
        #clipped_deltas = optimisers.clip_gradient([deltas_w_hy, deltas_w_hh, deltas_w_xh,
        #          deltas_b_hy, deltas_b_hh], 2)
        steps = self._optimiser(deltas, self.learning_rate)

        #steps = self._optimiser(clipped_deltas, self.learning_rate)

        for param, step in zip(params, steps):
            param -= step

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            learning_rate: float = 0.01,
            num_hidden_states: int = None,
            num_hidden_nodes: int = 5,
            num_backsteps: int = None,
            return_sequences: bool = False,
            independent_samples: bool = True,
            ) -> np.ndarray:
        """
        Method for training the RNN, iteratively runs _forward(), and
        _backwards() to predict values, find loss and adjust weights
        until a given number of training epochs has been reached.

        Parameters:
        -------------------------------
        X : np.array, shape: m x n
            - Input sequence, sequence elements may be scalars
              or vectors.
            - m: number of samples
            - n: number of features (for text, this corresponds to number
                                    ´of entries in embedding vector)
_
        y : np.array, shape: m x 1
            - Labels
            - m: equal to n in X    (for text, this corresponds to number
                                     of entries in embedding vector)

        epochs: int
            - Number of iterations to train for
                                    (1 epoch = iterate through all samples
                                     in X)

        learning_rate: float,

        num_hidden_states : int
            - Number of times to unroll the rnn architecture

        num_hidden_nodes : int
            - Number of fully connected hidden nodes to add

        num_backsteps : int
            - Number of hidden states to backpropagate through

        return_sequences : bool
            - Whether to return content of all output states (self.ys)
              or only the last output states. Shape:
              If True:
              shape = (num_hidden_states, time_steps, output_size)
              If False:
              shape = (num_hidden_states, output_size)

        return_sequences : bool
            - Whether to reset initial hidden state between each processed
              sample in X.

        Returns:
        -------------------------------
        (np.ndarray, np.ndarray) = (output states, hidden state)

        """

        X = np.array(X, dtype=object)  # object to allow inhomogeneous shape
        y = np.array(y, dtype=object)  # object to allow inhomogeneous shape

        samples, time_steps, num_features = X.shape
        samples, time_steps_y, output_size = y.shape

        self.learning_rate = learning_rate
        self.output_size = output_size
        self.num_features = num_features
        self.num_hidden_nodes = num_hidden_nodes

        if num_hidden_states is None:
            self.num_hidden_states = time_steps
        else:
            self.num_hidden_states = num_hidden_states

        self._init_weights()
        self._init_states()

        partitions = np.floor(time_steps/self.num_hidden_states)
        self.stats['loss'] = [0]*epochs

        if return_sequences:
            sequence_output = np.zeros((samples, time_steps, output_size))
        else:
            sequence_output = np.zeros((samples, output_size))

        for e in tqdm(range(epochs)):

            for sample in range(samples):

                if independent_samples:
                    self.hs[-1] = 0

                X_split = np.split(X[sample], partitions, axis=0)
                y_split = np.split(y[sample], partitions, axis=0)

                for X_partition, y_partition in zip(X_split, y_split):

                    y_pred = self._forward(
                        np.array(X_partition, dtype=float),
                        generate=False
                    )

                    if return_sequences:
                        sequence_output[sample] = self.ys
                    else:
                        sequence_output[sample] = self.ys[-1]

                    ret = self._loss(np.array(y_partition, dtype=int), self.ys, e)  #DTYPE INT!

                    self._backward()

        read_load_model.save_model(  # pickle dump the trained estimator
            self,
            'saved_models/',
            self.name
        )
        return sequence_output, self.hs[-1]

    def predict(
            self,
            x_seed: np.ndarray,
            h_seed: np.ndarray = None,
            time_steps_to_generate: int = 1,
            onehot: bool = False,
            ) -> np.ndarray:
        """
        Predicts the next value in a sequence of given inputs to the RNN
        network

        Parameters:
        -------------------------------
        x_seed : np.array
        - An X-sample to seed generation of samples

        h_seed : np.array
        - Hidden state value to seed generation of samples

        Returns:
        -------------------------------
        np.ndarray
        - Generated next samples

        """

        self.hs[-1] = h_seed
        self.num_hidden_states = time_steps_to_generate
        self._init_states()
        X = np.zeros((time_steps_to_generate, len(x_seed)))
        X[0] = x_seed
        self._forward(X, generate=True, onehot=onehot)

        return self.ys

    def _init_weights(
            self,
            scale=0.1) -> None:
        """
        Initialises weights and biases and assign them to instance variables.

        Parameters:
        -------------------------------
        scale : float
            - scaling of init weights
        Returns:
        -------------------------------
        None
        """
        # Notes:
        # w_xh = 1 x n, x = n x 1, => z = 1 x 1 (per state)
        # w_hh = hidden_layers x hidden_layers = 1 x 1, h = 1 x 1 (per state)
        # w_hy = 1 x hidden_layers = 1 x 1, y = 1 x 1 (per state)
        self.w_xh = np.random.randn(
            self.num_hidden_nodes, self.num_features) * scale
        self.w_hh = np.random.randn(
            self.num_hidden_nodes, self.num_hidden_nodes) * scale
        self.w_hy = np.random.randn(
            self.output_size, self.num_hidden_nodes) * scale

        self.b_hh = np.random.randn(
            self.num_hidden_nodes, self.num_hidden_nodes) * scale
        self.b_hy = np.random.randn(
            self.output_size, self.num_hidden_nodes) * scale

    def _init_states(self) -> None:
        """
        Initialises states and assign them to instance variables.

        Parameters:
        -------------------------------
        None
        Returns:
        -------------------------------
        None
        """

        if self.built:
            prev_h = self.hs[-1]
        self.hs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        if self.built:
            self.hs[-1] = prev_h
        self.xs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        self.ys = np.zeros((self.num_hidden_states, self.output_size))
        self.built = True

    def _loss(self, y_true, y_pred, epoch):
        loss = self._loss_function(y_true, y_pred)
        self.stats['loss'][epoch] += np.mean(loss)
        return loss

    def get_stats(self):
        return self.stats
