import sys
import git
import numpy as np
from rnn.rnn import RNN
from utils.activations import Relu, Tanh, Identity
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import save_model, load_model
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def onehot_encode(i, size_of_onehot):
    arr = np.zeros(size_of_onehot)
    arr[i] = 1
    return arr

story_lines = [
    "In the heart of the ancient forest, a mysterious glow emanated from beneath the verdant foliage.",
    "A lone traveler, weary from his journey, followed the ethereal light, hoping it would lead to sanctuary.",
    "The trees whispered secrets of old, guiding the traveler deeper into the forest's embrace.",
    "Suddenly, a clearing appeared, and in its center stood an ancient tree, its bark inscribed with runes of power.",
    "The traveler approached the tree, feeling a surge of energy coursing through his veins.",
    "From the shadows, forest creatures watched curiously, wondering if the traveler would unravel the forest's mysteries.",
    "As the traveler touched the tree, visions of ancient battles and forgotten lore flooded his mind.",
    "He realized he was not the first to seek the forest's wisdom, nor would he be the last.",
    "The creatures of the forest gathered, offering their aid to the traveler on his quest for knowledge.",
    "Together, they embarked on a journey to discover the secrets hidden within the depths of the mystical forest."
]
# Join the lines into a single string, separating sentences with a newline character for clarity.
# data = "\n".join(story_lines)

# data = open('input.txt', 'r').read()
import tensorflow as tf
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
data = open(path_to_file, 'rb').read().decode(encoding='utf-8')
chars = list(set(data))
print(chars)
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
ix_to_onehot = {i: onehot_encode(i, vocab_size) for i, ch in enumerate(chars)}
# print('char to ix:')
# print(char_to_ix)

# vocab skal være indeks til embedding = indeks til char som onehot
X = []
y = []
p = 0
seq_length = 50
num_inputs = 2000

for k in range(num_inputs):
    one_hot_x = np.zeros((seq_length, vocab_size))
    one_hot_y = np.zeros((seq_length, vocab_size))

    x_chars = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    y_chars = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    for i in range(len(x_chars)):
        idx_of_char = x_chars[i]
        one_hot_x[i][idx_of_char] = 1
    for i in range(len(y_chars)):
        idx_of_char = y_chars[i]
        one_hot_y[i][idx_of_char] = 1

    X.append(np.array(one_hot_x))
    y.append(np.array(one_hot_y))
    p += 1

train = True
infer = True
if train:

    epo = 1000
    hidden_nodes = 36
    # learning_rates = [0.001, 0.003, 0.005, 0.01]

    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Softmax()',
        loss_function='Classification_Logloss()',
        optimiser='AdaGrad()',
        clip_threshold=np.inf,
        name='tf_text_onehot_test_1',
        learning_rate=0.004,
        )

    hidden_state = rnn.fit(
        X,
        y,
        epo,
        num_hidden_nodes=hidden_nodes,
        return_sequences=True,
        independent_samples=True,
        num_backsteps=seq_length,
        vocab=ix_to_onehot,
        inverse_vocab=char_to_ix,
        )
    rnn.plot_loss(plt, show=True)

if infer:

    char = list('ROMEO:')
    X_seed = np.array([[ix_to_onehot[char_to_ix[c]] for c in char]])
    print(X_seed)

    print(X_seed.shape)
    rnn = load_model('saved_models/tf_text_onehot_test_1')
    # rnn.plot_loss(plt, show=True)
    predict = rnn.predict(X_seed, time_steps_to_generate=10)
    # print(predict)
    for emb in predict:
        print(ix_to_char[np.argmax(emb)])

# sample_ixs = []
# for r in ret[:-1]:
#     ix = np.argmax(r)
#     sample_ixs.append(ix)
# print(ix_to_char[np.argmax([X[0][0]])])
# txt = [ix_to_char[ix] for ix in sample_ixs]
# print(txt)
