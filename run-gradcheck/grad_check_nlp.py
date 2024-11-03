import sys
import git
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
import numpy as np
from rnn.rnn import RNN
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model

# ------ THIS FILE IS FOR RUNNING GRADIENT CHECK ------ #
gradcheck_at = 10  # check gradient at epoch 10

savepath = path_to_root + '/run-nlp/simple_cat/saved_models/simple_cat_model'

text_data = text_proc.read_file(path_to_root + '/data/embedding_test.txt')
chars = sorted(list(set(text_data)))  # to keep the order consistent over runs
data_size, vocab_size = len(text_data), len(chars)
print(f'Size: {data_size}, unique: {vocab_size}.')
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}


epo = 100
learning_rate = 0.0001
optimiser = 'Adam()'

seq_length = 6
num_samples = data_size-seq_length

X = np.zeros((num_samples, seq_length, 1, len(char_to_ix)))
y = np.zeros((num_samples, seq_length, 1, len(char_to_ix)))

for i in range(num_samples):
    inputs = [char_to_ix[ch] for ch in text_data[i:i + seq_length]]
    targets = [char_to_ix[ch] for ch in text_data[i + 1:i+seq_length+1]]
    onehot_x = text_proc.create_onehot(inputs, char_to_ix)
    onehot_y = text_proc.create_onehot(targets, char_to_ix)
    X[i] = onehot_x
    y[i] = onehot_y

X = X.transpose((2, 1, 0, 3))
y = y.transpose((2, 1, 0, 3))

print('Shape of X after batching:', X.shape)
print('Shape of y after batching:', y.shape)

rnn = RNN(
    hidden_activation='Tanh()',
    output_activation='Softmax()',
    loss_function='ce()',
    optimiser=optimiser,
    clip_threshold=1,
    learning_rate=learning_rate,
    seed=23
    )

rnn.fit(
    X,
    y,
    epo,
    num_hidden_nodes=10,
    vocab=ix_to_char,
    gradcheck_at=gradcheck_at,
)
