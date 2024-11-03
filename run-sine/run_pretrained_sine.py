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



########################################################################
########################### Sine creation ##############################
########################################################################

seq_length = 200
examples = 1

########################################################################
########################### Prediction #################################
########################################################################

seed_length = [10,10,3] #Seed lengths for each test

hidden_nodes = [10, 50] #Number of hidden nodes in the models to be retrieved
unrolling_steps = seq_length

learning_rates = [0.001,0.003,0.005,0.007,0.009] #Learning rates in the models 
                                                 #to be retrieved

#learning_rates = [0.004, 0.008]
#optimisers = ['AdaGrad()', 'SGD()', 'SGD_momentum()', 'Adam()']
optimisers = ['Adam()']
num_batches = 1

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


np.random.seed(13)
def create_random_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array(
            [np.random.uniform(-1, 1)*np.sin(
                np.random.uniform(-np.pi, np.pi)*np.linspace(0, 8*np.pi, seq_length+1))]
            ).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)


rng = np.random.default_rng(14)
def create_noisy_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array(
            [np.sin(rng.normal(0,0.6,seq_length + 1) +
                np.linspace(0, 8*np.pi, seq_length + 1))]
            ).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)

start_time = datetime.now()

for t in range(3):

    time_steps_to_predict = seq_length - seed_length[t]
    test = ['random', 'noisy', 'plain']
    if test[t] == 'random':
        X, y = create_random_sines(examples=examples, seq_length=seq_length)
    elif test[t] == 'noisy':
        X, y = create_noisy_sines(examples=examples, seq_length=seq_length)
    elif test[t] == 'plain':
        X, y = create_sines(examples=examples, seq_length=seq_length)

    for sine in X:
        plt.plot(sine[:])
    plt.show()

    torch_inputs = torch.tensor(X, dtype=torch.float32)
    torch_targets = torch.tensor(y, dtype=torch.float32)  # No squeeze needed
    torch_train_loader = DataLoader(
    TensorDataset(torch_inputs, torch_targets),
    batch_size=num_batches,
    shuffle=False
    )

    X = X.reshape(1, -1, num_batches, 1)
    y = y.reshape(1, -1, num_batches, 1)

    test = ['random', 'noisy', 'plain']

    if test[t] == 'random':
        X_val, y_val = create_random_sines(examples=examples, seq_length=seq_length)
    elif test[t] == 'noisy':
        X_val, y_val = create_noisy_sines(examples=examples, seq_length=seq_length)
    elif test[t] == 'plain':
        X_val, y_val = create_sines(examples=examples, seq_length=seq_length)
    
    #print(X_val.shape)
    #print(y_val.shape)

    torch_val_inputs = torch.tensor(X_val, dtype=torch.float32)
    torch_val_targets = torch.tensor(y_val, dtype=torch.float32)

    X_val = X_val.reshape(1, -1, 1, 1)
    y_val = y_val.reshape(1, -1, 1, 1)


    X_seed = np.array(X_val[0][:seed_length[t]])

    for num_hidden_nodes in hidden_nodes:
        for optimiser in optimisers:

            fig_pred = plt.figure()#figsize = (8, 7))
            fig_pred.suptitle(f'Predictions | Optimiser: {optimiser.split("()")[0]} | Hidden nodes: {num_hidden_nodes}')

            n_rows = int(np.ceil(len(learning_rates)/2))

            for learning_rate_curr, i in zip(learning_rates, range(len(learning_rates))):

                    rnn = load_model(f'{path_to_root}/run-sine/saved_models/pretrained_rnn_{test[t]}_{optimiser.split("()")[0]}_{num_hidden_nodes}_{learning_rate_curr}')
                    #print(X_seed.shape)
                    predict,y_seed_out = rnn.predict(X_seed,
                                        time_steps_to_generate=
                                        time_steps_to_predict,
                                        return_seed_out = True
                                        )
                    
                    predict = predict.squeeze()
                    y_seed_plot = np.array(y_seed_out).squeeze()
                    plot_line = np.concatenate((y_seed_plot, predict))

                    ax_pred = fig_pred.add_subplot(n_rows, 2, i + 1)
                    ax_pred.plot(plot_line - 3,
                                label=f'Numpy model')

                    ax_pred.set_xlabel("Time(t)")
                    ax_pred.axvline(x=seed_length[t]-1, color='black', linestyle='--')

                    torch_rnn = TORCH_RNN(
                                input_size=1,
                                hidden_size=num_hidden_nodes,
                                output_size=1
                            )
                    #torch_model_path = f'{path_to_root}/run-sine/saved_models/pretrained_torch_rnn_{test[t]}_{optimiser.split("()")[0]}_{learning_rate_curr}'
                    #print(torch_model_path)
                    torch_rnn.load_state_dict(torch.load(f'{path_to_root}/run-sine/saved_models/pretrained_torch_rnn_{test[t]}_{optimiser.split("()")[0]}_{num_hidden_nodes}_{learning_rate_curr}'))
                    
                    # Take the first sample of the training data for seeding
                    seed_data = torch_val_inputs[0:1,0:3]
                    seed_output, generated = torch_rnn.single_predict(seed_data, time_steps_to_predict + seed_length[t])
                    
                    torch_plot_line = np.concatenate((seed_data[0].squeeze(), generated))

                    ax_pred.plot(torch_plot_line - 6,
                                label=f'Torch model')

                    
                    y_plot_line = y_val[0].squeeze()
                    ax_pred.plot(y_plot_line, label="True")
                    ax_pred.set_title(f'Learning rate : {learning_rate_curr}')
                    handles, labels = ax_pred.get_legend_handles_labels()
                    fig_pred.legend(handles, labels, loc='lower right')

            fig_pred.set_layout_engine('tight')

            plt.show()
            #fig_pred.savefig(f'{path_to_root}/run-sine/saved_figs/pred_results_best_{optimiser.split("()")[0]}_{num_hidden_nodes}.svg')

            print(f'Execution time {datetime.now() - start_time}')


bytes_usage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
gb_usage_peak = round(bytes_usage_peak/1000000000, 3)
print('Memory consumption (peak):')
print(gb_usage_peak, 'GB')
