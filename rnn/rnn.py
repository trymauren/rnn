import sys
import os
import git
import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections.abc import Callable
from utils import optimisers
from utils import read_load_model
from utils.activations import Relu, Tanh, Identity, Sigmoid, Softmax
from utils.loss_functions import Mean_Square_Loss as mse
from utils.loss_functions import Classification_Logloss as ce
from utils.optimisers import SGD, SGD_momentum, AdaGrad, Adam


class RNN:

    def __init__(
            self,
            hidden_activation: Callable = None,
            output_activation: Callable = None,
            loss_function: Callable = None,
            optimiser: Callable = None,
            name: str = None,
            seed: int = 24,
            clip_threshold: float = 5,
            **optimiser_params,
            ) -> None:

        self.rng = np.random.default_rng(seed)

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

        self.name = name

        self.clip_threshold = clip_threshold

        self.stats = {
            'other stuff': [],
        }

        self.float_size = float

    def gradient_check(self, x, y, unrolling_steps, epsilon=1e-6):
        """ Performs gradient check for given x, y. """

        stored_states = self.states.copy()
        xs, hs, y_pred = self._forward(x)
        for x_state, h_state in zip(xs, hs):
            self.states.append((x_state, h_state))

        loss = self._loss_function(y, y_pred)

        while len(self.states) > unrolling_steps + 1:
            del self.states[0]

        analytical_grad = self._backward(check=True)

        self.states = stored_states.copy()

        count = 0
        param_names = ['U', 'W', 'V', 'b', 'c']
        print('GRADCHECK')
        for pidx, (param, pname) in enumerate(zip(self.parameters, param_names)):
            param_shape = param.shape
            numerical_grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index

                original_value = param[ix].copy()

                param[ix] = original_value + epsilon
                _, _, ys = self._forward(x)
                gradplus = self._loss_function(y, ys)
                # print('gradplus loss', gradplus)
                param[ix] = original_value - epsilon
                _, _, ys = self._forward(x)
                gradminus = self._loss_function(y, ys)
                # print('gradmins loss', gradminus)

                estimated_grad = (gradplus - gradminus)/(2*epsilon)

                param[ix] = original_value

                bptt_grad_param = analytical_grad[pidx][ix]

                # Calculating the relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(bptt_grad_param - estimated_grad)/(np.abs(bptt_grad_param) + np.abs(estimated_grad))

                if relative_error > 10*epsilon:
                    print(f'Epsilon:{epsilon}')
                    print(f"Gradient Check ERROR: parameter={pname}")
                    print(f"+h Loss: {gradplus}")
                    print(f"-h Loss: {gradminus}")
                    print(f"Estimated_gradient: {estimated_grad}")
                    print(f"Backpropagation gradient: {bptt_grad_param}")
                    print(f"Relative Error: {relative_error}")
                    print(f"Index of param in {pname} with error: {ix}, value of param: {param[ix]}")

                else:
                    print(f"Gradient check passed for parameter {pname}. Difference: {relative_error}")
                    print(f"Index of param in {pname} with correct value: {ix}, value of param: {param[ix]}")
                it.iternext()

    def _generate(self,
                  x,
                  time_steps_to_generate,
                  output_probabilities=False,
                  onehot=False
                  ):
        """
        This function (strictly) does not do generation, but rather
        predicts the next output value given an input value (x) and
        then feeds the output obtained as input at the next timestep.
        This cycling is what "generates" a sequence of values.

        Parameters:
        -------------------------------
        x : np.ndarray, shape: 1 x b x n
            - This means one timestep of a sequence
            - b: batchsize
            - n: number of features (for NLP, this corresponds to number
              of entries in embedding vector)

        time_steps_to_generate : int
            - How many new values to "generate"

        output_probabilities: bool
            - True if a "generated" value needs further processing
              before passed as input

        onehot: bool
            - True if a "generated" value needs further processing
              before passed as input

        Returns:
        -------------------------------
        output state: np.ndarray, shape: time_steps_to_generate x b x k
            - time_steps_to_generate: from argument to the parameter
            - b: batchsize
            - k: number of output nodes (output size)
        """

        ys = np.zeros((time_steps_to_generate,
                       self.batch_size,
                       self.num_features),
                      dtype=self.float_size)

        last_h = self.states[-1][1].copy()
        last_x = x

        for t in range(time_steps_to_generate):

            h, y = self._forward_unit(last_x, last_h)
            last_h = h

            self.states.append((last_x, h))  # this is not needed

            if output_probabilities:

                ix = self.prob_to_ix(y.flatten())

                if onehot:
                    outp = np.zeros((len(self.vocab), 1))
                    outp[ix] = 1
                    outp = outp.T

                else:
                    outp = self.ix_to_emb(ix)

                last_x = outp.copy()
                ys[t] = outp.copy()

            else:
                ys[t] = y
                last_x = y

        return ys

    def _forward_unit(self, x_step, last_h) -> None:
        """
        Executes the rnn cell equations

        Parameters:
        -------------------------------
        x_step : np.ndarray, shape: b x n
            - This means one timestep of a sequence
            - b: batchsize
            - n: number of features (for NLP, this corresponds to number
                                ´of entries in embedding vector)

        Returns:
        -------------------------------
        hidden state: np.ndarray, shape: b x h
            - b: batchsize
            - h: number of hidden nodes (hidden size)

        output state: np.ndarray, shape: b x k
            - b: batchsize
            - k: number of output nodes (output size)

        """

        x_weighted = x_step @ self.U.T
        h_weighted = last_h @ self.W.T
        a = self.b + h_weighted + x_weighted
        h = self._hidden_activation(a)
        o = self.c + h @ self.V.T
        y = self._output_activation(o)
        return h, y

    def _forward(self, x_sample) -> None:
        """
        Calls _forward_unit for amount of timesteps in x_sample and
        returns the states

        Parameters:
        -------------------------------
        x_sample : np.ndarray, shape: s x b x n
            - This means one batch of sequences
            - s: sequence length
            - b: batchsize
            - n: number of features (for NLP, this corresponds to number
                                ´of entries in embedding vector)

        Returns:
        -------------------------------
        input states: np.ndarray, shape: s x b x n
            - s: same as x_sample
            - b: same as x_sample
            - n: same as x_sample

        hidden states: np.ndarray, shape: s x b x h
            - s: same as x_sample
            - b: same as x_sample
            - h: number of hidden nodes in model

        output states: np.ndarray, shape: s x b x k
            - s: same as x_sample
            - b: same as x_sample
            - k: number of output nodes (output size)
        """

        xs = np.zeros((len(x_sample), self.batch_size, self.num_features))
        hs = np.zeros((len(x_sample), self.batch_size, self.num_hidden_nodes))
        ys = np.zeros((len(x_sample), self.batch_size, self.output_size))

        h = self.states[-1][1].copy()

        for t in range(len(x_sample)):
            h, y = self._forward_unit(x_sample[t], h)
            xs[t] = x_sample[t]
            hs[t] = h
            ys[t] = y

        return xs, hs, ys

    def _backward(self, check=False) -> None:
        """
        Calculates gradients with respect to each parameter, and calls
        the optimiser to get the step for each clipped gradient. The
        steps are then returned.

        Parameters:
        -------------------------------
        check: bool
            - True to return raw gradients (without clipping and steps)
              for use in gradient checking.

        Returns:
        -------------------------------
        steps: list containing np.ndarrays
            - The steps for each parameter
        """

        deltas_U = np.zeros_like(self.U, dtype=self.float_size)
        deltas_W = np.zeros_like(self.W, dtype=self.float_size)
        deltas_V = np.zeros_like(self.V, dtype=self.float_size)

        deltas_b = np.zeros_like(self.b, dtype=self.float_size)
        deltas_c = np.zeros_like(self.c, dtype=self.float_size)

        prev_grad_h_Cost = np.zeros((self.batch_size, self.num_hidden_nodes),
                                    dtype=self.float_size)

        loss_grad = self._loss_function.grad()

        for t in reversed(range(len(loss_grad))):

            grad_o_Cost = loss_grad[t]

            d_act = self._hidden_activation.grad(self.states[t+1][1])

            grad_h_Cost = grad_o_Cost @ self.V + prev_grad_h_Cost
            grad_h_Cost_raw = d_act * grad_h_Cost

            deltas_V += grad_o_Cost.T @ self.states[t+1][1]
            deltas_W += grad_h_Cost_raw.T @ self.states[t][1]
            deltas_U += grad_h_Cost_raw.T @ self.states[t+1][0]
            deltas_c += np.sum(grad_o_Cost, axis=0)
            deltas_b += np.sum(grad_h_Cost_raw, axis=0)
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
            unrolling_steps: int = 0,
            gradcheck_at: int = np.inf,
            vocab: dict = None,
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
        X : np.ndarray, shape: m x b x n
            - Input tensor
            - m: number of samples
            - s: length of each sample
            - b: batchsize
            - n: number of features (for NLP, this corresponds to number
                                    ´of entries in embedding vector)

        y : np.ndarray, shape: m x k
            - Target tensor
            - m: number of samples
            - s: length of each sample
            - b: batchsize
            - k: output size (for NLP, this corresponds to number
                              of entries in target vector)

        epochs: int
            - Number of iterations over all entries in X

        learning_rate: float

        num_hidden_nodes: int
            - Number of fully connected hidden nodes (hidden size)

        unrolling_steps: int
            - Number of hidden states to forward and backpropagate through

        vocab: dict,
            - If doing NLP, this dict has indices as keys and word-embeddings
              as values

        X_val: np.ndarray, shape: SAME as X
            - Passed if model should be validated during training

        y_val: np.ndarray, shape: SAME as y
            - Passed if model should be validated during training

        num_epochs_no_update: int
            - If X_val and y_val is passed, this value will determine when
              to stop training based on >0 change in validation loss

        Returns:
        -------------------------------
        hidden state: np.ndarray, shape: b x h
            - b: batchsize
            - h: number of hidden nodes (hidden size)

        """

        if gradcheck_at < epochs:
            self.float_size = np.float64

        if X.ndim != 4:
            raise ValueError('Input (X) must have 4 dimensions:\
                             Samples x sequence length x batches x features')
        if y.ndim != 4:
            raise ValueError('Output (y) must have 4 dimensions:\
                             Samples x sequence length x batches x output size')

        self.num_samples, seq_length, self.batch_size, self.num_features = X.shape

        self.num_samples, seq_length, self.batch_size, self.output_size = y.shape

        if seq_length > unrolling_steps:  # if unrolling_steps is not given
            unrolling_steps = seq_length

        self.num_hidden_nodes = num_hidden_nodes
        self._init_weights()
        self.vocab = vocab
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

                x_sample = np.asarray(x_sample, dtype=self.float_size)
                y_sample = np.asarray(y_sample, dtype=self.float_size)

                self._init_states()

                t_pointer = 0

                # Iterating over sub-sequence of the whole sequence, see
                # comment just below
                while t_pointer < seq_length:

                    # If the sequence is long (unrolling_steps) is given
                    # as argument, these lines picks out a range of
                    # timesteps to process in one forward/backward
                    # pass
                    self._dispatch_state(val=False)
                    pointer_end = t_pointer + unrolling_steps
                    pointer_end = min(pointer_end, seq_length)
                    x_batch = x_sample[t_pointer:pointer_end]
                    y_batch = y_sample[t_pointer:pointer_end]

                    # Not used in regular training, only to verify that
                    # gradients are correct
                    if e == gradcheck_at:
                        self.gradient_check(x_batch, y_batch, unrolling_steps)

                    xs, hs, y_pred = self._forward(x_batch)

                    # Store states from forward
                    for x, h in zip(xs, hs):
                        self.states.append((x, h))

                    # States not needed in backpropagation can be deleted
                    while len(self.states) > unrolling_steps + 1:
                        del self.states[0]

                    self._loss(y_batch, y_pred, e)

                    steps = self._backward()

                    # All code in this if-test is to process validation data
                    if self.val and len(X_val) > idx:
                        x_sample_val = X_val[idx]
                        y_sample_val = y_val[idx]
                        x_batch_val = x_sample_val[t_pointer:pointer_end]
                        y_batch_val = y_sample_val[t_pointer:pointer_end]
                        self._dispatch_state(val=self.val)
                        xs, hs, y_val_pred = self._forward(x_batch_val)
                        for x, h in zip(xs, hs):
                            self.states.append((x, h))
                        self._loss(y_batch_val, y_val_pred, e, val=self.val)

                    t_pointer += unrolling_steps

                    for param, step in zip(self.parameters, steps):
                        param -= step

                # processing of one sequence (or batch) while loop end
            # processing of a sample for loop end
        # epochs for loop end

            if self.val:
                if self.stats['val_loss'][e] >= self.stats['val_loss'][e-1]:
                    early_stop_counter += 1

                if early_stop_counter == num_epochs_no_update:
                    print(f'Val loss increasing, stopping fitting.')
                    break

            else:
                if self.stats['loss'][e] >= self.stats['loss'][e-1]:
                    early_stop_counter += 1

                if early_stop_counter == num_epochs_no_update:
                    print(f'Train loss increasing, stopping fitting.')
                    break

        if self.name is not None:
            read_load_model.save_model(self, self.name)

        self.stats['loss'] /= self.num_samples

        if self.val:
            self.stats['val_loss'] /= self.num_samples_val

        return self.ys, self.states[-1][1]

    def _loss(self, y_true, y_pred, epoch, val=False) -> None:
        """
        Calculates loss and stores it in statistics container. The loss
        calculated is also stored in the loss class, for use in
        gradient calculation
        """

        loss = self._loss_function(y_true, y_pred)
        self.stats['loss'][epoch] += loss
        if val:
            loss = self._loss_function(y_true, y_pred, nograd=True)
            self.stats['val_loss'][epoch] += loss

    def predict(
            self,
            X: np.ndarray,
            hs_init=None,
            time_steps_to_generate: int = 1,
            return_seed_out=False,
            onehot=False,
            ):
        """
        Predicts the next value(s) for a primer-sequence specified in X

        Parameters:
        -------------------------------
        X : np.ndarray, shape: s x 1 x n
            - An X-sample to seed generation of samples
            - n: number of features

        hs_init : bool
            - If True: use this hidden state as the last hidden state
            - If False: hidden state is initialised to zeros

        time_steps_to_generate: int
            - Number of timesteps to generate

        return_seed_out : bool
            - If True: see below
            - If False: None

        onehot : bool
            - Ideally, this parameter should not be present. However, as
              the code is now - there is need to distinguish between
              the character and word embedding cases

        Returns:
        -------------------------------
        Generated sequence as np.ndarray, shape: s x 1 x k
            - s: number of timesteps to generate, specified by
              time_steps_to_generate
            - k: output size

        Possibly the output states that are created as a result of
        forward-passing X, shape: s x 1 x k
            - s: same as X
            - k: number of output nodes (output size)
        """

        _, self.batch_size, num_features = X.shape

        xs_init = None
        if hs_init is None:
            hs_init = np.zeros((self.batch_size, self.num_hidden_nodes),
                               dtype=self.float_size)

        self.states = [(xs_init, hs_init)]

        xs, hs, seed_out = self._forward(X)

        last_seed_out = seed_out[-1, -1, :]

        for x_, h_ in zip(xs, hs):
            self.states.append((x_, h_))

        if self.vocab:

            # generating values
            if onehot:
                ix = self.prob_to_ix(last_seed_out)

                temp = np.zeros((1, len(last_seed_out)))

                temp[0, ix] = 1

                last_y_emb = temp

            else:
                last_y_emb = self.vocab[self.prob_to_ix(last_seed_out)]

            ys = self._generate(last_y_emb, time_steps_to_generate-1,
                                output_probabilities=True, onehot=onehot)

            if onehot:
                last_y_emb = last_y_emb.reshape(1, 1, last_y_emb.size)
                ys = np.concatenate((last_y_emb, ys))

            else:
                last_y_emb = last_y_emb.reshape(1, 1, len(last_y_emb))
                ys = np.concatenate((last_y_emb, ys))

            seed_out_ret = np.zeros((len(seed_out), self.batch_size,
                                     self.num_features))
            if return_seed_out:
                # loop over timesteps in seed-returns
                for t in range(len(seed_out_ret)):  # in range timesteps
                    emb = self.ix_to_emb(self.prob_to_ix(seed_out[t, -1, :]))
                    seed_out_ret[t] = emb

                return ys, seed_out_ret

        else:

            # generating values
            ys = self._generate(last_seed_out, time_steps_to_generate-1,
                                output_probabilities=False)

            # add the last seed output(this is the first truly generated value)
            last_seed_out = last_seed_out.reshape(1, 1, len(last_seed_out))
            ys = np.concatenate((last_seed_out, ys))

            seed_out_ret = np.zeros((len(seed_out), self.batch_size, self.num_features))
            if return_seed_out:
                # loop over timesteps in seed-returns
                for t in range(len(seed_out_ret)):  # in range timesteps
                    out = seed_out[t, -1, :]
                    seed_out_ret[t] = out
                return ys, seed_out_ret

        return ys

    def ix_to_emb(self, index):
        """
        Returns the word embedding or one-hot representation of a
        character corresponding to the index in the vocabulary
        """

        embedding = self.vocab[index]
        return embedding

    def prob_to_ix(self, probabilities):
        """
        Samples an index from probabilities parameter using the models
        specified rng generator
        """

        return self.rng.choice(range(len(probabilities)), p=probabilities)

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
        self.U = self.rng.uniform(
            -0.3, 0.3, size=(self.num_hidden_nodes,
                             self.num_features
                             )
            )
        self.W = self.rng.uniform(
            -0.3, 0.3, size=(self.num_hidden_nodes,
                             self.num_hidden_nodes
                             )
            )
        self.V = self.rng.uniform(
            -0.3, 0.3, size=(self.output_size,
                             self.num_hidden_nodes
                             )
            )

        self.b = self.rng.uniform(
            -0.3, 0.3, size=(
                             self.num_hidden_nodes
                             )
            )
        self.c = self.rng.uniform(
            -0.3, 0.3, size=(self.output_size
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
        'Dispatches' the states of validation/training by swapping out
         what self.states references. If val is True, then validation
         states are dispatched
        """

        if not val:
            self.val_states = self.states.copy()
            self.states = self.train_states

        if val:
            self.train_states = self.states.copy()
            self.states = self.val_states

    def _init_states(self):
        """
        Initialises the states memory.
        Parameters:
        -------------------------------
        None

        Returns:
        -------------------------------
        None
        """
        xs_init = None
        hs_init = np.zeros((self.batch_size, self.num_hidden_nodes),
                           dtype=self.float_size)
        init_states = [(xs_init, hs_init)]
        self.train_states = init_states

        if self.val:
            self.val_states = self.train_states.copy()

        self.states = self.train_states

    def plot_loss(self, plt, figax=None, savepath=None, show=False, val=False):

        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        ax.set_yscale('symlog')
        # ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.plot(
                self.stats['loss'],
                label='Numpy model',
                alpha=1,
                linestyle='solid')
        if val:
            if self.val:
                ax.plot(
                        self.stats['val_loss'],
                        label='Numpy model val',
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
        """
        Fetches the saved statistics as a dictionary

        The keys are currently:
            - 'loss'
            - 'val_loss' (if validation set is passed to fit())
            - 'parameter_count'
            - 'parameters'
        """

        return self.stats
