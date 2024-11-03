import sys
import git
import numpy as np
import matplotlib.pyplot as plt
from rnn.rnn import RNN as RNN
from rnn.rnn import RNN as RNN_parallel
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array(
            [np.sin(
                np.linspace(0, 4*np.pi, seq_length+1))]
            ).T
        # example_x = np.repeat(example_x, 2, axis=1)
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


seq_length = 30
examples = 1
epo = 100
hidden_nodes = 30
num_backsteps = 30
num_forwardsteps = 30
learning_rate = 0.001
optimiser = 'Adam()'
num_batches = 1
features = 1

X, y = create_sines(examples=examples, seq_length=seq_length)

X_batched = X.reshape(examples, -1, num_batches, features)
y_batched = y.reshape(examples, -1, num_batches, features)
X_nonbatched = X.reshape(examples, num_batches, -1, features)
y_nonbatched = y.reshape(examples, num_batches, -1, features)


# rnn = RNN(
#     hidden_activation='Tanh()',
#     output_activation='Identity()',
#     loss_function='mse()',
#     optimiser=optimiser,
#     clip_threshold=1,
#     learning_rate=learning_rate,
#     )

# hidden_state = rnn.fit(
#     X_nonbatched,
#     y_nonbatched,
#     epo,
#     num_hidden_nodes=hidden_nodes,
#     num_backsteps=num_backsteps,
#     num_forwardsteps=num_backsteps,
# )
# plt.plot(rnn.get_stats()['loss'], label='rnn')



rnn_batch = RNN_parallel(
    hidden_activation='Tanh()',
    output_activation='Identity()',
    loss_function='mse()',
    optimiser=optimiser,
    clip_threshold=1,
    learning_rate=learning_rate,
    )

hidden_state_batch = rnn_batch.fit(
    X_batched,
    y_batched,
    epo,
    num_hidden_nodes=hidden_nodes,
    # num_backsteps=num_backsteps,
    # num_forwardsteps=num_forwardsteps,
    # gradcheck_at=10,
    # X_val=X_batched,
    # y_val=y_batched,
)
seed = X_batched[0, :10, :, :]
ret = rnn_batch.predict(seed, time_steps_to_generate=20)

plt.plot(rnn_batch.get_stats()['loss'], label='batch train')
# plt.plot(rnn_batch.get_stats()['val_loss'], label='batch val')

plt.legend()
plt.show()

plt.plot(np.concatenate((seed.squeeze(), ret.squeeze())))
plt.plot(y_batched.squeeze())
plt.show()