import sys
import os
import git
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class RNN_1d_regression(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN_1d_regression, self).__init__()

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            nonlinearity='tanh')

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state=None):
        if h_state is not None:
            hidden_out, h_state = self.rnn(x, h_state)
        else:
            hidden_out, h_state = self.rnn(x)
        out = self.fc(hidden_out)
        return out, h_state

    def fit(self, train_loader, epochs=5, lr=0.01):

        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.parameters(), lr=lr)  # Could use Adam
        self.loss_list = np.zeros(epochs)

        for e in tqdm(range(epochs)):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs, _ = self(inputs)
                loss = criterion(outputs, targets)
                self.loss_list[e] = loss
                loss.backward()
                optimizer.step()

    def single_predict(self, seed_data, timesteps):

        self.eval()

        with torch.no_grad():

            output, h_state = self(seed_data)  # seed the model (h_state)
            seed_output = output.flatten().numpy()  # for plotting only
            #print(seed_output)
            last_output = output[0][-2:-1]  # extract output at last time-step
            loop_h_state = h_state[0]
            # the weird slice above is to get correct dimensions
            print(h_state.shape)
            generated_data = []

            # ret.append(float(last_output))

            for t in range(timesteps):
                last_output, h_state = self(last_output,h_state=loop_h_state)
                loop_h_state = h_state
                generated_data.append(float(np.squeeze(last_output)))

        return seed_output, generated_data


if __name__ == "__main__":

    # ------------ Data creation ------------ #
    def create_sines(examples=10, seq_length=20):
        X, y = [], []
        for _ in range(examples):
            example_x = np.sin(np.linspace(0, 8*np.pi, seq_length+1))
            # Reshape for single feature per timestep
            X.append(example_x[:-1].reshape(-1, 1))
            y.append(example_x[1:].reshape(-1, 1))
        return np.array(X), np.array(y)

    inputs, targets = create_sines(examples=1, seq_length=200)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)  # No squeeze needed
    train_loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=1,
        shuffle=False
        )

    # ------------ Config ------------ #
    train = True
    infer = True
    num_seed_values = 3
    hidden_size = 10
    epochs = 300
    learning_rate = 0.001

    # ------------ Model definition ------------ #
    model = RNN_1d_regression(
        input_size=1,
        hidden_size=hidden_size,
        output_size=1
        )

    # ------------ Train ------------ #
    if train:
        model.fit(train_loader, epochs=epochs, lr=learning_rate)
        with open('./rnn/loss_list.pkl', 'wb') as file:
            pickle.dump(model.loss_list, file)
        plt.figure()
        plt.plot(model.loss_list)
        plt.title('Loss over epochs')
        plt.show()
        torch.save(model.state_dict(), './rnn/torch_sine')

    # ------------ Inference ------------ #
    if infer:
        model = RNN_1d_regression(
            input_size=1,
            hidden_size=hidden_size,
            output_size=1
            )
        model.load_state_dict(torch.load('./rnn/torch_sine'))
        # Take the first sample of the training data for seeding
        seed_data = inputs[0:1]
        seed_output, generated = model.single_predict(seed_data, 200)
        concatenated = np.concatenate((seed_output, generated))
        plt.plot(seed_data[0], label='Should have been')
        plt.plot(concatenated, label='generated values')
        plt.axvline(x=num_seed_values, ls='--',
                    label='seeded to mark, generated after')
        plt.legend()

        plt.figure()
        with open('./rnn/loss_list.pkl', 'rb') as file:
            data = pickle.load(file)
        plt.plot(data)
        plt.title('Loss over epochs')
        plt.show()
