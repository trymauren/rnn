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
from utils.loss_functions import Classification_Logloss
from utils import optimisers
from utils.optimisers import SGD, SGD_momentum, AdaGrad, RMSProp
from utils import read_load_model
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class RNN:

    def __init__(
            self,
            hidden_activation: Callable = None,
            output_activation: Callable = None,
            loss_function: Callable = None,
            optimiser: Callable = None,
            name: str = 'rnn',
            config: Dict | Path | str = 'default',
            seed: int = 24,
            clip_threshold: float = 5,
            **optimiser_params,
            ) -> None:

        np.random.seed(seed)

        # Setting activation functions, loss function and optimiser
        if not hidden_activation:
            hidden_activation = Tanh()
        self._hidden_activation = eval(hidden_activation)
        if not output_activation:
            output_activation = Identity()
        self._output_activation = eval(output_activation)

        if not loss_function:
            loss_function = mse()
        self._loss_function = eval(loss_function)

        if not optimiser:
            optimiser = AdaGrad()
        self._optimiser = eval(optimiser)
        self.optimiser_params = optimiser_params

        # Initialize weights and biases as None until properly
        # initialized in fit() method.
        # xh = input  -> hidden
        # hh = hidden -> hidden
        # hy = hidden -> output
        self.U, self.W, self.V = None, None, None

        self.b, self.c = None, None

        self.xs, self.hs, self.ys = None, None, None

        self.built = False

        self.name = name

        self.clip_threshold = clip_threshold

        self.stats = {
            'other stuff': [],
        }

    def _forward(
            self,
            x_sample,
            generate=False,
            nograd=False,
            output_probabilities=False,
            num_forwardsteps=0,
            t_pointer=0,
            ) -> None:
        """
        Forward-pass method to be used in fit-method for training the
        RNN. Returns predicted output values

        Parameters:
        -------------------------------
        x_sample:
            - A sample of vectors

        generate:
            - Whether to insert output at time t=1 as input at time t=2

        Returns:
        -------------------------------
        None
        """
        def onehot_to_embedding(index):
            embedding = self.vocab[index]
            return embedding

        xs = np.zeros_like(self.xs)
        hs = np.zeros_like(self.hs)
        ys = np.zeros_like(self.ys)

        if not num_forwardsteps:
            num_forwardsteps = len(x_sample)
            print('num_forwardsteps was 0')

        if num_forwardsteps + t_pointer > len(x_sample):
            num_forwardsteps = len(x_sample) - t_pointer

        for t in range(t_pointer, num_forwardsteps):
            x_weighted = self.U @ x_sample[t]
            h_weighted = self.W @ self.hs[t-1]
            a = self.b + h_weighted + x_weighted
            xs[t] = a
            h = self._hidden_activation(a)
            hs[t] = h
            o = self.c + self.V @ hs[t]
            y = self._output_activation(o)
            ys[t] = y

            if not nograd:
                self.xs[t] = xs[t]
                self.hs[t] = hs[t]
                self.ys[t] = ys[t]

            if generate:
                if t < self.num_hidden_states - 1:
                    if output_probabilities:
                        ix = self.probabilities_to_index(ys[t])
                        x_sample[t+1] = onehot_to_embedding(ix)
                    else:
                        x_sample[t+1] = ys[t]

        return ys

    def _backward(self, num_backsteps=np.inf, t_pointer=None) -> None:
        debug = True
        deltas_U = np.zeros_like(self.U, dtype=float)
        deltas_W = np.zeros_like(self.W, dtype=float)
        deltas_V = np.zeros_like(self.V, dtype=float)

        deltas_b = np.zeros_like(self.b, dtype=float)
        deltas_c = np.zeros_like(self.c, dtype=float)

        prev_grad_h_Cost = np.zeros_like(self.num_hidden_nodes)

        loss_grad = self._loss_function.grad()

        # num_backsteps = min(len(self.hs)-1, num_backsteps)
        # if not t_pointer - num_backsteps:
        #     num_backsteps = 0  # may change to +1/-1
        # if debug:
        #     print('lossgrad shape:', loss_grad.shape)
        #     print('t_pointer backward:', t_pointer)
        #     print('numbacksteps backward:', num_backsteps)

        for t in range(t_pointer-1, t_pointer-1 - num_backsteps, -1):
            # print('hbefifwefwihhihbifwhih')
            # print('t,', t)
            if debug:
                print('t_pointer backward:', t_pointer)
                print('numbacksteps backward:', num_backsteps)
                print('t in backward:', t)
            """ BELOW IS CALCULATION OF GRADIENTS W/RESPECT TO HIDDEN_STATES """
            grad_o_Cost_t = loss_grad[t]
            # grad_h_Cost = optimisers.clip_gradient(grad_h_Cost, self.clip_threshold)
            """A h_state's gradient update are both influenced by the
            preceding h_state at time t+1, as well as the output at
            time t. The cost/loss of the current output derivated with
            respect to hidden state t is what makes up the following
            line before the "+ sign". After "+" is the gradient through
            previous hidden states and their outputs. This term after
            the "+" sign, is 0 for first step of BPTT.

            Eq. 16 in tex-document(see also eq. 15 for first iteration of BPPT)
            Eq. 10.20 in DLB"""
            grad_h_Cost = prev_grad_h_Cost + self.V.T @ grad_o_Cost_t

            # print(prev_grad_h_Cost.shape)
            # print(self.V.T.shape)
            # print(grad_h_Cost.shape)
            """The following line is to shorten equations. It fetches/
            differentiates the hidden activation function."""
            d_act = self._hidden_activation.grad(self.hs[t])

            """ BELOW IS CALCULATION OF GRADIENT W/RESPECT TO WEIGHTS """
            # print(np.array([self.hs[t]]).shape)
            # print(grad_o_Cost_t.shape)
            # exit()
            """Cumulate the error."""
            # print(grad_o_Cost_t * self.hs[t] == grad_o_Cost_t @ np.array([self.hs[t]]))
            deltas_V += grad_o_Cost_t * self.hs[t]  # 10.24 in DLB
            deltas_W += d_act @ self.hs[t-1] * grad_h_Cost  # 10.26 in DLB
            print('d_act:', d_act.shape); print('xs[t]:', self.xs[t].shape); print('grad_h_cost:', grad_h_Cost.shape)
            print('deltas_U:', deltas_U.shape)
            print()
            deltas_U += d_act @ self.xs[t].T * grad_h_Cost  # 10.28 in DLB
            deltas_c += grad_o_Cost_t.T * 1  # 10.22 in DLB
            deltas_b += d_act @ grad_h_Cost  # 10.22 in DLB

            """Pass on the bits of the chain rule to the calculation of
            the previous hidden state update
            This line equals the first part of eq. 10.21 in DLB
            To emphasize: the part before the "+" in 10.21 in DLB"""
            prev_grad_h_Cost = d_act @ self.W.T @ grad_h_Cost

        params = [self.V, self.W, self.U,
                  self.c, self.b]
        deltas = [deltas_V, deltas_W, deltas_U,
                  deltas_c, deltas_b]

        clipped_deltas = optimisers.clip_gradient(deltas, self.clip_threshold)
        # steps = self._optimiser(deltas, **self.optimiser_params)

        steps = self._optimiser(clipped_deltas, **self.optimiser_params)

        # for param, step in zip(params, steps):
        #     param -= step
        return steps

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            num_hidden_nodes: int = 5,
            num_forwardsteps: int = 0,
            num_backsteps: int = 0,
            return_sequences: bool = False,
            independent_samples: bool = True,
            vocab=None,
            inverse_vocab=None,
            ) -> np.ndarray:
        #  X.shape should be NUM_SEQUENCES x BATCH_SIZE x SEQ_LENGTH x NUM_FEATURES?
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
        self.vocab, self.inverse_vocab = vocab, inverse_vocab
        X = np.array(X, dtype=object)  # object to allow inhomogeneous shape
        y = np.array(y, dtype=object)  # object to allow inhomogeneous shape

        if X.ndim != 4:
            raise ValueError('Input data for X has to be of 3 dimensions:\
                             Samples x num_batches x time steps x features')
        if y.ndim != 4:
            raise ValueError('Input data for y has to be of 3 dimensions:\
                             Samples x num_batches x time steps x features')

        if not num_forwardsteps:
            num_forwardsteps = np.inf

        _, num_batches, seq_length, num_features = X.shape
        _, num_batches, seq_length, output_size = y.shape

        self.output_size = output_size
        self.num_features = num_features
        self.num_hidden_nodes = num_hidden_nodes

        self._init_weights()

        self.stats['loss'] = np.zeros(epochs)
        debug = False

        for e in range(epochs):

            for sample_x, sample_y in zip(X, y):

                sample_x = np.array(sample_x, dtype=float)  # asarray()?
                sample_y = np.array(sample_y, dtype=float)  # asarray()?

                # OK!
                self.num_hidden_states = seq_length
                self.batch_xs = np.zeros((num_batches, self.num_hidden_states,
                                          self.num_hidden_nodes))
                self.batch_hs = np.zeros((num_batches, self.num_hidden_states,
                                          self.num_hidden_nodes))
                self.batch_ys = np.zeros((num_batches, self.num_hidden_states,
                                          self.output_size))

                for batch_ix in range(num_batches):  # OK!
                    self._init_states()
                    self.batch_xs[batch_ix] = self.xs
                    self.batch_hs[batch_ix] = self.hs
                    self.batch_ys[batch_ix] = self.ys

                t_pointer = 0

                while t_pointer < len(sample_x):

                    batch_steps = [0]*num_batches

                    for batch_ix in range(num_batches):

                        # Dispatch hidden state of the current batch
                        self.dispatch_state(batch_ix=batch_ix)
                        x_batch = sample_x[batch_ix]
                        y_batch = sample_y[batch_ix]
                        # # The following two lines could be utilised
                        # # to avoid batching as pre-processing
                        # start_ix = batch_ix * num_forwardsteps
                        # endat_ix = start_ix + num_forwardsteps

                        y_pred = self._forward(
                            x_batch,
                            num_forwardsteps=num_forwardsteps,
                            t_pointer=t_pointer
                        )

                        self._loss(y_batch, y_pred,
                                   t_pointer, num_forwardsteps, e)

                        t_pointer += num_forwardsteps

                        steps = self._backward(
                            num_backsteps=num_backsteps,
                            t_pointer=t_pointer
                        )
                        batch_steps[batch_ix] = steps

                    average_step = np.mean(np.array(batch_steps, dtype=object), axis=0)

                    params = [self.V, self.W, self.U,
                              self.c, self.b]

                    for param, step in zip(params, average_step):
                        param -= step

        read_load_model.save_model(
            self,
            'saved_models/',
            self.name
        )
        print("Training complete, proceed")
        return self.ys, self.hs[-1]

    def predict(
            self,
            X: np.ndarray,
            time_steps_to_generate: int = 1,
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

        if X.ndim > 3 or X.ndim < 2:
            raise ValueError("Input data for X has to be of 3 dimensions:\
                             Samples x time steps x features")
        if X.ndim == 3:
            X = X[0]  # remove first dim of X

        _, num_features = X.shape
        X_gen = np.zeros((time_steps_to_generate, num_features))

        self.num_hidden_states = len(X)
        self._init_states()

        ys = self._forward(np.array(X, dtype=float))

        if self.vocab:
            last_y_emb = self.vocab[self.probabilities_to_index(ys[-1])]
            X_gen[0] = last_y_emb
            self.num_hidden_states = time_steps_to_generate
            last_h = self.hs[-1]
            self._init_states()
            self.hs[-1] = last_h
            ys = self._forward(X_gen, generate=True,
                               output_probabilities=True)
            return [self.vocab[self.probabilities_to_index(y)] for y in ys]

        else:
            X_gen[0] = ys[-1]
            self.num_hidden_states = time_steps_to_generate
            last_h = self.hs[-1]
            self._init_states()
            self.hs[-1] = last_h
            ys = self._forward(X_gen, generate=True,
                               output_probabilities=False)
            return ys

    def probabilities_to_index(self, probabilities):
        return np.random.choice(range(len(probabilities)), p=probabilities)

    def _init_weights(self) -> None:
        """
        Initialises weights and biases and assign them to instance variables.

        Parameters:
        -------------------------------
        None
        Returns:
        -------------------------------
        None
        """
        self.U = np.random.uniform(
            -0.3, 0.3, size=(self.num_hidden_nodes, self.num_features))
        self.W = np.random.uniform(
            -0.3, 0.3, size=(self.num_hidden_nodes, self.num_hidden_nodes))
        self.V = np.random.uniform(
            -0.3, 0.3, size=(self.output_size, self.num_hidden_nodes))

        self.b = np.random.uniform(
            -0.3, 0.3, size=(1, self.num_hidden_nodes))
        self.c = np.random.uniform(
            -0.3, 0.3, size=(1, self.output_size))

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
        self.hs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        self.xs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        self.ys = np.zeros((self.num_hidden_states, self.output_size))

    def dispatch_state(self, batch_ix) -> None:
        """
        'Dispatches' the states of current batch by swapping the
        reference of self._s variables.
        Parameters:
        -------------------------------
        None
        Returns:
        -------------------------------
        None
        """
        self.xs = self.batch_xs[batch_ix]
        self.hs = self.batch_hs[batch_ix]
        self.ys = self.batch_ys[batch_ix]

    def _loss(self, y_true, y_pred, t_pointer, num_forwardsteps, epoch):

        t_pointer_end = t_pointer + num_forwardsteps

        loss = self._loss_function(y_true[t_pointer:t_pointer_end],
                                   y_pred[t_pointer:t_pointer_end])

        self.stats['loss'][epoch] += np.mean(loss)

    def plot_loss(self, plt, figax=None, savepath=None, show=False):
        # Some config stuff
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        ax.set_yscale('symlog')
        ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.plot(
                self.stats['loss'],
                label=str(self._optimiser.__class__.__name__))

        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training loss')
        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
        return fig, ax

    def get_stats(self):
        return self.stats