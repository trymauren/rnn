import numpy as np
import sys
import git
path_to_root = git.Repo('../', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
from rnn.rnn import RNN


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

X, y = create_sines(seq_length=100)

X = X.reshape(1, -1, 1, 1)
y = y.reshape(1, -1, 1, 1)
print(X.shape)

rnn = RNN(
    hidden_activation='Tanh()',
    output_activation='Identity()',
    loss_function='mse()',
    optimiser='Adam()',
    clip_threshold=1,
    learning_rate=0.001,
    )
hidden_state = rnn.fit(
    X,
    y,
    epochs=60,
    num_hidden_nodes=100,
    unrolling_steps=100,
    gradcheck_at=60
)




