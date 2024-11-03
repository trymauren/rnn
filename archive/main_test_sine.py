import sys
import git
import resource
import numpy as np
# from rnn.rnn import RNN
from rnn.rnn_batch_new import RNN
# from lstm.lstm import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
from utils.loss_functions import Mean_Square_Loss
from datetime import datetime
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        # example_x = np.array(
        #     [np.random.uniform(-2, 2)*np.sin(
        #         np.linspace(0, (seq_length/7.5)*np.pi, seq_length+1))]
        #     ).T
        example_x = np.array(
            [np.sin(
                np.linspace(0, 8*np.pi, seq_length+1))]
            ).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)


def create_cosines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array(
            [np.random.uniform(-2, 2)*np.cos(
                np.linspace(0, 4*np.pi, seq_length+1))]
            ).T
        example_x = np.array(
            [np.cos(
                np.linspace(0, 8*np.pi, seq_length+1))]
            ).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)


# def error(prediction, validation) -> float:
#     mse = Mean_Square_Loss()

#     print(prediction.shape)
#     print(validation.shape)
#     exit()
#     accuracy = mse.eval(validation, prediction)
#     return np.round(accuracy[0], 3)


# def model_performance_sine(rnn, num_tests, test_length, seed_length = 10, seq_length = 100):
#     """
#     Parameters:
#     ------------------------------------------------------
#     rnn:
#         - rnn model to measure performance on.
#     num_tests:
#         - number of predictions to base performance measure upon
#     test_length:
#         - length of each prediction
#     seed_length:
#         - length of input/seed to base a prediction upon
#     seq_length:
#         - length of the created sine waves used as data

#     Returns:
#     -------------------------------------------------------
#     Average mean square error between predicted and actual sine waves
#     across several predictions: 
#         - float
#     """
#     accuracies = []
#     X_val, y_val = create_sines(num_tests, seq_length)
#     max_start = 100 - test_length
#     random_start = np.random.randint(0, max_start)
#     time_steps_to_predict = test_length - seed_length
#     fig, ax = plt.subplots()
#     for sine in X_val:
#         X_seed = np.array([sine[random_start:random_start + seed_length]])
#         predict = rnn.predict(X_seed,
#                               time_steps_to_generate=time_steps_to_predict)
#         accuracies.append(error(predict, sine[random_start
#                           + seed_length:random_start + test_length]))
#         ax.plot(sine[random_start + seed_length:random_start + test_length],
#                 label="True")
#         ax.plot(predict, label="Predict", linestyle='dotted')
#         plt.legend()
#     return np.round(np.mean(accuracies), 3)


# Sine creation
seq_length = 60
examples = 1

# Prediction
seed_length = 3
time_steps_to_predict = seq_length - seed_length

# RNN init
epo = 1000
hidden_nodes = 40  # < 40 hidden_nodes is not able to capture periodicity
# for lr=o.oo1, Adagrad, epo=4000, seq_length=30, num_backsteps=30
num_backsteps = 30
# learning_rates = [0.001,0.003,0.005,0.007,0.009]
learning_rates = [0.001]
# optimisers = ['AdaGrad()', 'SGD()', 'SGD_momentum()', 'RMSProp()']
optimisers = ['AdaGrad()']
num_batches = 2

# Plotting
offset = 3

start_time = datetime.now()

X, y = create_sines(examples=examples, seq_length=seq_length)

# Plotting the sine waves that are passed as training data
# plt.title("Randomized sines used for training")
# plt.ylabel("Amplitude(y)")
# plt.xlabel("Time(t)")

# for sine in X:
#     plt.plot(sine)
# plt.savefig(f'Sine_training_data | size = {examples}')

# X = X.reshape(1, 5, 20, 1)
# y = y.reshape(1, 5, 20, 1)
# X = X.reshape(1, 20, -1, 1)
# y = y.reshape(1, 20, -1, 1)
# X = X.reshape(1, 5, -1, 1)
# y = y.reshape(1, 5, -1, 1)
X = X.reshape(examples, -1, num_batches, 1)
y = y.reshape(examples, -1, num_batches, 1)


X_val, y_val = create_sines(examples=1, seq_length=seq_length)

X_seed = np.array([X_val[0][:seed_length]])
print(X_seed.shape)
X_seed = X_seed.reshape(-1, 1, 1)
# X_val_batch = X_val.reshape(1, num_batches, -1, 1)
# y_val_batch = y_val.reshape(1, num_batches, -1, 1)

for learning_rate_curr in learning_rates:
    print(f'\n\n---------------------\nlearning rate: {learning_rate_curr}')
    print('---------------------')

    fig_loss, ax_loss = plt.subplots()
    fig_pred, ax_pred = plt.subplots()

    ax_pred.set_title(f"Predictions | learning rate: {learning_rate_curr}")
    ax_pred.set_yticks([])
    ax_pred.set_xlabel("Time(t)")
    ax_pred.axvline(x=seed_length-1, color='black', linestyle='--')

    y_ticks_pos = [2]
    y_ticks_acc = ['Error']

    for optimiser, i in zip(optimisers, range(len(optimisers))):
        print(f'Training model with {optimiser.split("()")[0]}')
        rnn = RNN(
            hidden_activation='Tanh()',
            output_activation='Identity()',
            loss_function='mse()',
            optimiser=optimiser,
            clip_threshold=1,
            learning_rate=learning_rate_curr,
            )

        hidden_state = rnn.fit(
            X,
            y,
            epo,
            num_hidden_nodes=hidden_nodes,
            num_backsteps=num_backsteps,
            num_forwardsteps=num_backsteps,
            # X_val=X_val_batch,
            # y_val=y_val_batch,
        )
        s = f'Sine_train_loss_{learning_rate_curr}.svg'
        rnn.plot_loss(plt, figax=(fig_loss, ax_loss), show=False, val=True)

        predict = rnn.predict(X_seed,
                              time_steps_to_generate=time_steps_to_predict)
        predict = predict.flatten()
        X_seed_plot = X_seed.flatten()
        plot_line = np.concatenate((X_seed_plot, predict))
        ax_pred.plot(plot_line - (i+1)*offset,
                     label=str(rnn._optimiser.__class__.__name__))
        # y_ticks_pos.append( - (i+1)*offset)
        # y_ticks_acc.append(f'{error(predict, X_val[0][seed_length:])}')


        # performance = model_performance_sine(rnn, 10, 30)
        # print(f'Average prediction error: {performance}\n')

    ax_pred.plot(X_val[0], label="True")
    ax_pred.legend()
    ax_pred.set_yticks(y_ticks_pos, labels=y_ticks_acc)

    #print(accuracy())
    # ax_loss.set_title(f"Loss | learning rate: {learning_rate_curr}")

    # fig_loss.savefig(f'Sine_train_loss_{learning_rate_curr}.svg')
    # fig_pred.savefig(f'Sine_pred_{learning_rate_curr}.svg')
    print(f'Execution time {datetime.now() - start_time}')

    plt.show()

bytes_usage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
gb_usage_peak = round(bytes_usage_peak/1000000000, 3)
print('Memory consumption (peak):')
print(gb_usage_peak, 'GB')
