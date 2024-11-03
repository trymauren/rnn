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


class TORCH_RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(TORCH_RNN, self).__init__()

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

    def fit(self, train_loader, optimizer, epochs=5, lr=0.01):

        if optimizer == 'AdaGrad()':
            optimizer = optim.Adagrad(self.parameters(), lr=lr)
        elif optimizer == 'SGD()':
            optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'SGD_momentum()':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer == 'RMSProp()':
            optimizer = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'Adam()':
            optimizer = optim.Adam(self.parameters(), lr=lr)

        criterion = nn.MSELoss(reduction='mean')
        optimizer = optimizer  # Could use Adam
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
            #for i in range(len(seed_data)):                                             #FAULTY ATTEMPT AT SEEDING MORE THAN ONE TIMESTEP
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