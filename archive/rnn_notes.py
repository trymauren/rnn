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

        self.stats = {}

    def _forward(
            self,
            x_sample,
            generate=False,
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
        for t in range(self.num_hidden_states):
            x_weighted = self.U @ x_sample[t]
            h_weighted = self.W @ self.hs[t-1]
            a = self.b + x_weighted + h_weighted
            self.xs[t] = a
            h = self._hidden_activation(a)
            self.hs[t] = h
            o = self.c + self.V @ self.hs[t]
            self.ys[t] = self._output_activation(o)
            if generate:
                if t < self.num_hidden_states - 1:
                    x_sample[t+1] = self.ys[t]

        return self.ys

    def _backward(self, num_backsteps=np.inf) -> None:

        deltas_U = np.zeros_like(self.U, dtype=float)
        deltas_W = np.zeros_like(self.W, dtype=float)
        deltas_V = np.zeros_like(self.V, dtype=float)

        deltas_b = np.zeros_like(self.b, dtype=float)
        deltas_c = np.zeros_like(self.c, dtype=float)

        prev_grad_h_Cost = np.zeros_like(self.num_hidden_nodes)

        #NOTE: Implemented gradient clipping, however shape error, 
        #      gradient norm is a list of floats, not one number
        loss_grad = self._loss_function.grad()

        num_backsteps = min(len(self.hs)-1, num_backsteps)
        for t in range(num_backsteps, -1, -1):

            """ BELOW IS CALCULATION OF GRADIENTS W/RESPECT TO HIDDEN_STATES """
            grad_o_Cost_t = loss_grad[:, t]
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

            """Cumulate the error."""
            deltas_V += self.hs[t].T * grad_o_Cost_t  # 10.24 in DLB
            deltas_W += d_act @ self.hs[t-1] * grad_h_Cost  # 10.26 in DLB
            deltas_U += d_act @ self.xs[t] * grad_h_Cost  # 10.28 in DLB
            deltas_c += grad_o_Cost_t.T * 1  # 10.22 in DLB
            deltas_b += d_act @ grad_h_Cost  # 10.22 in DLB

            """Pass on the bits of the chain rule to the calculation of
            the previous hidden state update
            This line equals the first part of eq. 10.21 in DLB
            To emphasize: the part before the "+" in 10.21 in DLB"""
            prev_grad_h_Cost = d_act @ self.W.T @ grad_h_Cost
            prev_grad_h_Cost = prev_grad_h_Cost

        params = [self.V, self.W, self.U,
                  self.c, self.b]
        deltas = [deltas_V, deltas_W, deltas_U,
                  deltas_c, deltas_b]
        clipped_deltas = optimisers.clip_gradient(deltas, self.clip_threshold)
        # steps = self._optimiser(deltas, **self.optimiser_params)

        steps = self._optimiser(clipped_deltas, **self.optimiser_params)

        for param, step in zip(params, steps):
            param -= step

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            num_hidden_nodes: int = 5,
            num_backsteps: int = None,
            return_sequences: bool = False,
            independent_samples: bool = True,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
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
        (np.ndarray, np.ndarray) =
        (output states of last seen sample, hidden state of last epoch)
        """

        X = np.array(X, dtype=object)  # object to allow inhomogeneous shape
        y = np.array(y, dtype=object)  # object to allow inhomogeneous shape

        if X.ndim != 3:
            raise ValueError("Input data for X has to be of 3 dimensions:\
                             Samples x time steps x features")
        if y.ndim != 3:
            raise ValueError("Input data for y has to be of 3 dimensions:\
                             Samples x time steps x features")
        print("Please wait, training model:")

        _, _, num_features = X.shape
        _, _, output_size = y.shape

        self.output_size = output_size
        self.num_features = num_features
        self.num_hidden_nodes = num_hidden_nodes

        self._init_weights()

        self.stats['train_loss'] = np.zeros(epochs)
        if X_val is not None and y_val is not None:
            self.stats['val_loss'] = np.zeros(epochs)

        for e in tqdm(range(epochs)):

            # looping over each sequence in dataset
            for idx, (sample_x, sample_y) in enumerate(zip(X, y)):

                self.num_hidden_states = len(sample_x)

                # re-initialise states for each sample
                self._init_states(independent_samples=independent_samples)

                # one forward pass of all entries in sample_x
                y_pred = self._forward(
                    np.array(sample_x, dtype=float),
                    generate=False
                )

                # calculate loss
                self._loss(np.array(sample_y, dtype=float), self.ys, e, validation=False)
                # print('train hs')
                # print(self.hs[-1])
                if X_val is not None and y_val is not None:

                    # Store everything
                    self._store_states()
                    self.num_hidden_states = len(X_val[idx])
                    self._init_states(
                        independent_samples=independent_samples,
                        train=False)
                    # print('val hs')
                    # print(self.hs[-1])

                    # one forward pass of all entries in X_val[idx]
                    y_pred = self._forward(
                        np.array(X_val[idx], dtype=float),
                        generate=False
                    )

                    # calculate loss
                    self._loss(np.array(y_val[idx], dtype=float), y_pred, e, validation=True)

                    # self._val_hs = np.copy(self.hs[-1]) # uncomment!!
                    # print('val hs')
                    # print(self.hs[-1])
                    # Restore
                    self._restore_states()
                # print('train hs')
                # print(self.hs[-1])

                # one backward pass for each entry in x_sample (or less,
                # if num_backsteps is set to < len(x_sample))
                self._backward(num_backsteps=num_backsteps)

            # reset previous hidden state each epoch
            self.hs[-1] = np.zeros_like(self.hs[-1])

        read_load_model.save_model(  # pickle dump the trained estimator
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
        # if h_seed is None:
        #     self.hs[-1] = np.zeros_like(self.hs[-1])
        # else:
        #     self.hs[-1] = h_seed
        if X.ndim != 3:
            raise ValueError("Input data for X has to be of 3 dimensions:\
                             Samples x time steps x features")

        self.num_hidden_states = time_steps_to_generate
        self._init_states()
        # X = np.zeros((time_steps_to_generate, len(x_seed)))
        # X[0] = x_seed
        _,_,vec_length = X.shape
        X_gen = np.zeros((time_steps_to_generate, vec_length))
        X_gen[0] = X[-1][-1]
        for x in X[:-1]:
            self._forward(np.array(x, dtype=float))
        self._forward(np.array(X_gen, dtype=float), generate=True)
        return self.ys

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

    def _init_states(self, independent_samples=True, train=True) -> None:
        """
        Initialises states and assign them to instance variables. If
        independent samples is set, hs[-1] is not preserved.

        Parameters:
        -------------------------------
        None
        Returns:
        -------------------------------
        None
        """
        if train:
            if independent_samples:
                self._init_states_zero()

            else:
                print('WRONG')
                exit()
                if self.built:
                    prev_h = np.copy(self.hs[-1])
                    self._init_states_zero()
                    self.hs[-1] = prev_h
                else:
                    self._init_states_zero()
                    self.built = True

        else:
            if independent_samples:
                self._init_states_zero()

            else:
                print('WRONG')
                exit()
                if self.built_val:
                    prev_h = np.copy(self.hs[-1])
                    self._init_states_zero()
                    self.hs[-1] = prev_h
                else:
                    self._init_states_zero()
                    self.built_val = True

    def _init_states_zero(self):
        self.hs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        self.xs = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
        self.ys = np.zeros((self.num_hidden_states, self.output_size))

    # def _init_states_zero_val(self):
    #     self.hs_val = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
    #     self.xs_val = np.zeros((self.num_hidden_states, self.num_hidden_nodes))
    #     self.ys_val = np.zeros((self.num_hidden_states, self.output_size))

    def _store_states(self):
        self.xs_train = np.copy(self.xs)
        self.hs_train = np.copy(self.hs)
        self.ys_train = np.copy(self.ys)

    def _restore_states(self):
        self.xs_val = np.copy(self.xs)
        self.hs_val = np.copy(self.hs)
        self.ys_val = np.copy(self.ys)
        self.xs = self.xs_train
        self.hs = self.hs_train
        self.ys = self.ys_train

    def _loss(self, y_true, y_pred, epoch, validation=False):
        """
        Calculates loss using self._loss_function() and stores loss in
        self.stats['{val}/{train}_loss']

        Parameters:
        -------------------------------
        y_true : np.ndarray
        - The label of an x_sample

        y_pred : np.ndarray
        - The predicted value of an x_sample

        epoch : integer
        - For correct placement in statistics-dict

        Returns
        -------------------------------
        None
        """
        if validation:
            loss = self._loss_function(y_true, y_pred, nograd=True)
            self.stats['val_loss'][epoch] += np.mean(loss)

        else:
            loss = self._loss_function(y_true, y_pred)
            self.stats['train_loss'][epoch] += np.mean(loss)


    def plot_loss(self, plt, figax=None, savepath=None, show=False):
        # Some config stuff
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        # ax.set_yscale('symlog')
        # ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
        # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.plot(
                self.stats['train_loss'],
                label='train')

        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Train loss')
        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
        return fig, ax

    def plot_loss_val(self, plt, figax=None, savepath=None, show=False):
        # Some config stuff
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        # ax.set_yscale('symlog')
        # ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
        # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.plot(
                self.stats['val_loss'],
                label='val')

        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Validation loss')
        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
        return fig, ax


import sys
import git
import numpy as np
from rnn.rnn import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array([np.random.uniform(-2,2)*np.sin(np.linspace(0, 4*np.pi, seq_length+1))]).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)

def create_cosines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array([np.sin(np.linspace(0, 4*np.pi, seq_length+1))]).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)

# Sine creation
seq_length = 50
examples = 100

# Prediction
seed_length = 2
time_steps_to_predict = seq_length - seed_length

# RNN init
epo = 1000
hidden_nodes = 100
num_backsteps = seq_length
learning_rates = [0.005]
# learning_rates = [0.001,0.003,0.005,0.007,0.009]
optimisers = ['AdaGrad()']
# optimisers = ['AdaGrad()', 'SGD()', 'SGD_momentum()','RMSProp()']

# Plotting
offset = 3

X, y = create_sines(examples=examples, seq_length=seq_length)

# # Plotting the sine waves that are passed as training data
# plt.title("Randomized sines used for training")
# plt.ylabel("Amplitude(y)")
# plt.xlabel("Time(t)")

# for sine in X:
#     plt.plot(sine)
# plt.show()

X_val, y_val = create_cosines(examples=examples, seq_length=seq_length)
X_seed = np.array([X_val[0][:seed_length]])


for learning_rate_curr in learning_rates:
    print(f'learning rate: {learning_rate_curr}')

    fig_loss, ax_loss = plt.subplots()
    # fig_pred, ax_pred = plt.subplots()

    fig_loss_val, ax_loss_val = plt.subplots()
    # fig_pred_val, ax_pred_val = plt.subplots()

    # ax_pred.set_title(f"Predictions | learning rate: {learning_rate_curr}")
    # ax_pred.set_yticks([])
    # ax_pred.set_xlabel("Time(t)")
    # ax_pred.axvline(x=seed_length-1, color='black', linestyle='--')

    for optimiser, i in zip(optimisers, range(len(optimisers))):
        rnn = RNN(
            hidden_activation='Tanh()',
            output_activation='Identity()',
            loss_function='mse()',
            optimiser=optimiser,
            clip_threshold=1,
            learning_rate=learning_rate_curr,
            )

        hidden_state = rnn.fit(
            X, y, epo,
            num_hidden_nodes=hidden_nodes, return_sequences=True,
            num_backsteps=num_backsteps, X_val=X_val, y_val=y_val)

        rnn.plot_loss(plt, figax=(fig_loss, ax_loss), show=False)
        rnn.plot_loss_val(plt, figax=(fig_loss, ax_loss), show=False)

        # predict = rnn.predict(X_seed, time_steps_to_generate=time_steps_to_predict)

        # plot_line = np.concatenate((X_seed[0],predict))
        # ax_pred.plot(plot_line - (i+1)*offset, label=str(rnn._optimiser.__class__.__name__))
        # ax_pred.legend()
    # ax_pred.plot(X_val[0], label = "X val")
    # ax_pred.legend()
plt.show()