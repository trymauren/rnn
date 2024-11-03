import sys
import os
import git
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from collections.abc import Callable
from utils import optimisers
from utils import read_load_model
from utils.activations import Relu, Tanh, Identity, Softmax
from utils.loss_functions import Mean_Square_Loss as mse
from utils.loss_functions import Classification_Logloss
from utils.optimisers import SGD, SGD_momentum, AdaGrad, RMSProp
import jax
from jax import jit
import jax.numpy as jnp
# path_to_root = git.Repo('.', search_parent_directories=True).working_dir
# sys.path.append(path_to_root)
import operator

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
        # U = input  -> hidden
        # W = hidden -> hidden
        # V = hidden -> output
        self.U, self.W, self.V = None, None, None

        self.b, self.c = None, None

        self.xs, self.hs, self.ys = None, None, None

        self.built = False

        self.name = name

        self.clip_threshold = clip_threshold

        self.stats = {
            'other stuff': [],
        }

        self.float_size = np.float64

    def gradient_check(self, x, y, epsilon=1e-6):

        stored_states = self.states.copy()
        xs, hs, y_pred = self._forward(x)
        for x_state, h_state in zip(xs, hs):
            self.states.append((x_state, h_state))

        self._loss_function(y, y_pred)

        analytical_grad = self._backward(check=True)

        self.states = stored_states.copy()

        count = 0
        param_names = ['U', 'W', 'V', 'b', 'c']

        for pidx, (param, pname) in enumerate(zip(self.parameters, param_names)):
            param_shape = param.shape
            numerical_grad = np.zeros_like(param)

            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index

                original_value = param[ix].copy()

                param[ix] = original_value + epsilon
                _, _, ys = self._forward(x)
                gradplus = self._loss_function(ys, y)

                param[ix] = original_value - epsilon
                _, _, ys = self._forward(x)
                gradminus = self._loss_function(ys, y)

                estimated_grad = (gradplus - gradminus)/(2*epsilon)

                param[ix] = original_value

                bptt_grad_param = analytical_grad[pidx][ix]

                # Calculating the relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(bptt_grad_param - estimated_grad)/(np.abs(bptt_grad_param) + np.abs(estimated_grad))

                if relative_error > epsilon:
                    print(f'Epsilon:{epsilon}')
                    print(f"Gradient Check ERROR: parameter={pname}")
                    print(f"+h Loss: {gradplus}")
                    print(f"-h Loss: {gradminus}")
                    print(f"Estimated_gradient: {estimated_grad}")
                    print(f"Backpropagation gradient: {bptt_grad_param}")
                    print(f"Relative Error: {relative_error}")
                    exit()
                else:
                    print(f"Gradient check passed for parameter {pname}. Difference: {relative_error}")
                it.iternext()

    def jax_shit_up(self):
        self._forward_unit = jit(self._forward_unit)
        self._forward = jit(self._forward_jax)

    def _generate(self, X_seed, output_probabilities=False):
        def onehot_to_embedding(index):
            embedding = self.vocab[index]
            return embedding

        ys = np.zeros((len(X_seed), self.batch_size, self.output_size), dtype=self.float_size)

        for t in range(len(X_seed)):
            a, h, o, y = self._forward_unit(X_seed[t], self.states[t][1])
            self.states.append((None, h))
            ys[t] = y
            if t < len(X_seed)-1:
                if output_probabilities:
                    ix = self.probabilities_to_index(y.flatten())
                    X_seed[t+1] = onehot_to_embedding(ix)
                else:
                    X_seed[t+1] = y
        return ys

    def _forward_unit(self, x_sample, last_h):
        x_weighted = x_sample @ self.U.T
        h_weighted = last_h @ self.W.T # last hidden state ## DENNE MÅ SJEKKES!! Skal W transponeres? Problem pga n x n matrise
        a = self.b.T + h_weighted + x_weighted
        h = self._hidden_activation(a)
        o = self.c.T + h @ self.V.T
        y = self._output_activation(o)

        return a, h, o, y

    def _forward(
            self,
            x_sample,
            nograd=False,
            output_probabilities=False,
            debug=False,
            ) -> None:

        xs = np.zeros((len(x_sample), self.batch_size, self.num_features))
        hs = np.zeros((len(x_sample), self.batch_size, self.num_hidden_nodes))
        ys = np.zeros((len(x_sample), self.batch_size, self.output_size))

        h = self.states[-1][1].copy()

        for t in range(len(x_sample)):
            a, h, o, y = self._forward_unit(x_sample[t], h)
            xs[t] = x_sample[t]
            hs[t] = h
            ys[t] = y

        return xs, hs, ys

    def _forward_jax(
            self,
            x_sample,
            nograd=False,
            output_probabilities=False,
            debug=False,
            ) -> None:

        def onehot_to_embedding(index):
            embedding = self.vocab[index]
            return embedding

        xs = jnp.zeros((len(x_sample), self.batch_size, self.num_features))
        hs = jnp.zeros((len(x_sample), self.batch_size, self.num_hidden_nodes))
        ys = jnp.zeros((len(x_sample), self.batch_size, self.output_size))

        h = self.states[-1][1].copy()

        for t in range(len(x_sample)):
            a, h, o, y = self._forward_unit(x_sample[t], h)
            xs.at[t].set(x_sample[t])
            hs.at[t].set(h)
            ys.at[t].set(y)

        return xs, hs, ys

    def _backward(self, check=False) -> None:
        debug = True
        deltas_U = np.zeros_like(self.U, dtype=self.float_size)
        deltas_W = np.zeros_like(self.W, dtype=self.float_size)
        deltas_V = np.zeros_like(self.V, dtype=self.float_size)

        deltas_b = np.zeros_like(self.b, dtype=self.float_size)
        deltas_c = np.zeros_like(self.c, dtype=self.float_size)

        prev_grad_h_Cost = np.zeros((self.batch_size, self.num_hidden_nodes), dtype=self.float_size)

        loss_grad = self._loss_function.grad()

        for t in reversed(range(len(loss_grad) + 1)):
            if self.states[t][0] is None:
                break  # reached init state

            grad_o_Cost_t = loss_grad[t-1]

            d_act = 1 - np.square(self.states[t][1])

            grad_h_Cost = grad_o_Cost_t @ self.V + prev_grad_h_Cost
            grad_h_Cost_raw = d_act * grad_h_Cost

            deltas_V += grad_o_Cost_t.T @ self.states[t][1]/self.batch_size
            deltas_W += grad_h_Cost_raw.T @ self.states[t-1][1]/self.batch_size
            deltas_U += grad_h_Cost_raw.T @ self.states[t][0]/self.batch_size
            deltas_c += np.sum(grad_o_Cost_t, axis=0) * 1/self.batch_size
            deltas_b += np.sum(grad_h_Cost_raw, axis=0, keepdims=True).T/self.batch_size
            prev_grad_h_Cost = grad_h_Cost_raw @ self.W

        deltas = [deltas_U, deltas_W, deltas_V,
                  deltas_b, deltas_c]

        if check:
            return deltas

        clipped_deltas = optimisers.clip_gradient(deltas, self.clip_threshold)

        steps = self._optimiser(clipped_deltas, **self.optimiser_params)

        return steps

    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None,
            epochs: int = None,
            num_hidden_nodes: int = 5,
            num_forwardsteps: int = 0,
            num_backsteps: int = 0,
            gradcheck_at: int = np.inf,
            vocab=None,
            inverse_vocab=None,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            num_epochs_no_update: int = None,
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
        (np.ndarray, np.ndarray) = (output states, hidden state)

        """
        # self.jax_shit_up()

        X = np.array(X, dtype=object)  # object to allow inhomogeneous shape
        y = np.array(y, dtype=object)  # object to allow inhomogeneous shape

        if X.ndim != 4:
            raise ValueError('Input (X) must have 4 dimensions:\
                             Samples x batches x time steps x features')
        if y.ndim != 4:
            raise ValueError('Output (y) must have 4 dimensions:\
                             Samples x batches x time steps x features')

        self.num_samples, seq_length, self.batch_size, num_features = X.shape
        self.num_samples, seq_length, self.batch_size, output_size = y.shape

        if not num_forwardsteps:
            print('Warning: number of forwarsteps is not specified - \
                processing the whole sequence. This may use some memory \
                for long sequences')
            num_forwardsteps = seq_length

        self.output_size = output_size
        self.num_features = num_features
        self.num_hidden_nodes = num_hidden_nodes
        self._init_weights()
        self.vocab, self.inverse_vocab = vocab, inverse_vocab
        self.stats['loss'] = np.zeros(epochs)
        self.val = False

        if X_val is not None and y_val is not None:
            self.val = True
            self.stats['val_loss'] = np.zeros(epochs)
            self.num_samples_val = X_val.shape[0]

        early_stop_counter = 0

        for e in tqdm(range(epochs)):

            self.e = e

            for idx, (x_sample, y_sample) in enumerate(zip(X, y)):

                x_sample = np.array(x_sample, dtype=self.float_size)
                y_sample = np.array(y_sample, dtype=self.float_size)

                self._init_states()

                t_pointer = 0

                while t_pointer < seq_length:

                    self._dispatch_state(val=False)

                    pointer_end = t_pointer + num_forwardsteps
                    pointer_end = min(pointer_end, seq_length)

                    x_batch = x_sample[t_pointer:pointer_end]
                    y_batch = y_sample[t_pointer:pointer_end]

                    if e == gradcheck_at:
                        self.gradient_check(x_batch, y_batch)

                    xs, hs, y_pred = self._forward(x_batch)

                    for x, h in zip(xs, hs):
                        self.states.append((x, h))

                    while len(self.states) > num_backsteps + 1:
                        del self.states[0]

                    self._loss(y_batch, y_pred, e)

                    steps = self._backward()

                    if self.val and len(X_val) > idx:
                        x_sample_val = X_val[idx]
                        y_sample_val = y_val[idx]
                        x_batch_val = x_sample_val[t_pointer:pointer_end]
                        y_batch_val = y_sample_val[t_pointer:pointer_end]
                        self._dispatch_state(val=self.val)
                        xs, hs, y_val_pred = self._forward(x_batch_val, nograd=True)
                        for x, h in zip(xs, hs):
                            self.states.append((x, h))
                        self._loss(y_batch_val, y_val_pred, e, val=self.val)

                    t_pointer += num_forwardsteps

                    for param, step in zip(self.parameters, steps):
                        param -= step

            if self.val:
                if self.stats['val_loss'][e] >= self.stats['val_loss'][e-1]:
                    early_stop_counter += 1
                if early_stop_counter == num_epochs_no_update:
                    print(f'Val loss increasing, stopping fitting.')
                    break

        read_load_model.save_model(self, 'saved_models/', self.name)

        self.stats['loss'] /= self.num_samples

        if self.val:
            self.stats['val_loss'] /= self.num_samples_val

        print('Train complete')

        return self.ys, self.states[-1][1]

    def _loss(self, y_true, y_pred, epoch, val=False):
        loss = self._loss_function(y_true, y_pred)
        self.stats['loss'][epoch] += loss
        if val:
            loss = self._loss_function(y_true, y_pred, nograd=True)
            self.stats['val_loss'][epoch] += loss

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
        X : np.array
        - An X-sample to seed generation of samples

        Returns:
        -------------------------------
        np.ndarray
        - Generated sequence
        """

        # if X.ndim > 3 or X.ndim < 2:
        #     raise ValueError("Input data for X has to be of 3 dimensions:\
        #                      Samples x time steps x features")

        _, self.batch_size, num_features = X.shape

        X_gen = np.zeros((time_steps_to_generate, self.batch_size, num_features), dtype=self.float_size)

        xs_init = None
        hs_init = np.full((self.batch_size, self.num_hidden_nodes), 0, dtype=self.float_size)
        self.states = [(xs_init, hs_init)]

        xs, hs, seed_out = self._forward(X, debug=True)
        for x_, h_ in zip(xs, hs):
            self.states.append((x_, h_))
        if self.vocab:
            last_y_emb = self.vocab[self.probabilities_to_index(seed_out[-1, -1, :])]
            X_gen[0] = last_y_emb
            ys = self._generate(X_gen, output_probabilities=True)
            return [self.vocab[self.probabilities_to_index(y.flatten())] for y in ys]

        else:
            X_gen[0] = seed_out[-1][-1]  # this may not be robust
            ys = self._generate(X_gen, output_probabilities=False)
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
            -0.3, 0.3, size=(self.num_hidden_nodes,
                             self.num_features
                             )
            )
        self.W = np.random.uniform(
            -0.3, 0.3, size=(self.num_hidden_nodes,
                             self.num_hidden_nodes
                             )
            )
        self.V = np.random.uniform(
            -0.3, 0.3, size=(self.output_size,
                             self.num_hidden_nodes
                             )
            )

        self.b = np.random.uniform(
            -0.3, 0.3, size=(
                             self.num_hidden_nodes, 1
                             )
            )
        self.c = np.random.uniform(
            -0.3, 0.3, size=(1, self.output_size
                             )
            )

        self.parameters = [self.U, self.W, self.V,
                           self.b, self.c]

        total = sum(np.size(param) for param in self.parameters)
        self.stats['parameter_count'] = total
        self.stats['parameters'] = {'U:', self.U.shape,
                                    'W:', self.W.shape,
                                    'V:', self.V.shape,
                                    'b:', self.b.shape,
                                    'c:', self.c.shape}

    def _dispatch_state(self, val=False) -> None:
        """
        'Dispatches' the states of current batch by swapping out what
        self.states references.
        Parameters:
        -------------------------------
        None
        Returns:
        -------------------------------
        None
        """
        if not val:
            self.val_states = self.states.copy()
            self.states = self.train_states

        if val:
            self.train_states = self.states.copy()
            self.states = self.val_states

    def _init_states(self):

        xs_init = None
        hs_init = np.full((self.batch_size, self.num_hidden_nodes), 0, dtype=self.float_size)
        init_states = [(xs_init, hs_init)]
        self.train_states = init_states

        if self.val:
            self.val_states = self.train_states.copy()

        self.states = self.train_states

    def plot_loss(self, plt, figax=None, savepath=None, show=False, val=False):
        # Some config stuff
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        ax.set_yscale('symlog')
        # ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.plot(
                self.stats['loss'],
                label=str(self._optimiser.__class__.__name__),
                alpha=1,
                linestyle='solid')
        if val:
            if self.val:
                ax.plot(
                        self.stats['val_loss'],
                        label=str(self._optimiser.__class__.__name__) + '_val',
                        alpha=1,
                        linestyle='dotted')
            else:
                print('Model has not been validated - creating plot without \
                       validation data')

        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Loss over epochs')

        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
        return fig, ax

    def get_stats(self):
        return self.stats
