import sys
import git
path_to_root = git.Repo('../', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
import resource
import numpy as np
import os
#sys.path.append(os.path.abspath('..'))
from rnn.rnn import RNN
from rnn.pytorch_rnn_sine import TORCH_RNN
# from lstm.lstm import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
from utils.loss_functions import Mean_Square_Loss
from utils.read_load_model import load_model
from datetime import datetime
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm



########################################################################
########################### Sine creation ##############################
########################################################################

seq_length = 200
examples = 10

########################################################################
########################### Parameters #################################
########################################################################

seed_length = 10
time_steps_to_predict = seq_length - seed_length

epo = 2
hidden_nodes = 40
unrolling_steps = seq_length

learning_rate = 0.001


optimiser = 'Adam()'

num_batches = 1

num_models = 75

########################################################################
########################## Script start ################################
########################################################################

def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array(
            [np.sin(
                np.linspace(0, 8*np.pi, seq_length+1))]
            ).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)

X, y = create_sines(examples=examples, seq_length=seq_length)

torch_inputs = torch.tensor(X, dtype=torch.float32)
torch_targets = torch.tensor(y, dtype=torch.float32)  # No squeeze needed
torch_train_loader = DataLoader(
TensorDataset(torch_inputs, torch_targets),
batch_size=num_batches,
shuffle=False
)

X = X.reshape(1, -1, num_batches, 1)
y = y.reshape(1, -1, num_batches, 1)


print("\n\n##########################################################\n\
##       Starting training speed comparison test        ##\n\
##########################################################")

print("\n\n#--------------------------------------------------------#\n\
#                  Training Numpy model                  #\n\
#--------------------------------------------------------#")

numpy_times = []
start_time_numpy = datetime.now().timestamp()

for i in range(num_models):
    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Identity()',
        loss_function='mse()',
        optimiser=optimiser,
        clip_threshold=1,
        learning_rate=learning_rate,
        )
    numpy_fit_start = datetime.now().timestamp()
    hidden_state = rnn.fit(
        X,
        y,
        epo,
        num_hidden_nodes=hidden_nodes,
        unrolling_steps=unrolling_steps,
    )
    numpy_fit_time = datetime.now().timestamp() - numpy_fit_start
    numpy_times.append(numpy_fit_time)

execution_time_numpy = datetime.now().timestamp() - start_time_numpy
average_numpy = np.mean(numpy_times)


print("\n\n#--------------------------------------------------------#\n\
#                 Training PyTorch model                 #\n\
#--------------------------------------------------------#")

pytorch_times = []
start_time_pytorch = datetime.now().timestamp()

for i in range(num_models):

    torch_rnn = TORCH_RNN(
                input_size=1,
                hidden_size=hidden_nodes,
                output_size=1,
                )
    
    pytorch_fit_start = datetime.now().timestamp()
    torch_rnn.fit(torch_train_loader, epochs=epo, 
                    lr=learning_rate, optimizer = optimiser)
    
    pytorch_fit_time = datetime.now().timestamp() - pytorch_fit_start
    pytorch_times.append(pytorch_fit_time)

execution_time_pytorch = datetime.now().timestamp() - start_time_pytorch
average_pytorch = np.mean(pytorch_times)


print('\n\n-----------------------------')
print(f'Total training time for all {num_models} numpy models: {np.round(execution_time_numpy,3)}s')
print(f'Average Numpy model training time: {np.round(average_numpy, 3)}s')
print(f'\nTotal training time for all {num_models} PyTorch models: {np.round(execution_time_pytorch,3)}s')
print(f'Average PyTorch model training time: {np.round(average_pytorch, 3)}s')
print(f'\nTotal training time difference for all models: {np.round(np.abs(execution_time_numpy - execution_time_pytorch),3)}s')
print(f'Average training time difference per model: {np.round(np.abs(average_numpy - average_pytorch),3)}s')

if execution_time_numpy < execution_time_pytorch:
     winner = 'Numpy'
     percentage = (1 - (execution_time_numpy/execution_time_pytorch))*100
else:
     winner = 'PyTorch'
     percentage = (1 -(execution_time_pytorch/execution_time_numpy))*100

print(f'\nThe {winner} model was in total {np.round(percentage,3)}% faster!')
print('-----------------------------')

plt.show()
bytes_usage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
gb_usage_peak = round(bytes_usage_peak/1000000000, 3)
print('Memory consumption (peak):')
print(gb_usage_peak, 'GB')
