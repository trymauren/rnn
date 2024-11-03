import sys
import git
import numpy as np
import matplotlib.pyplot as plt
from rnn.rnn import RNN as RNN_parallel
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model
import tensorflow as tf

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

epo = 5
hidden_nodes = 1200
num_backsteps = 25
num_forwardsteps = 25
learning_rate = 0.001
optimiser = 'AdaGrad()'
num_batches = 16

word_emb = WORD_EMBEDDING()
text_data = text_proc.read_file("data/three_little_pigs.txt")

path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text_data = open(path_to_file, 'rb').read().decode(encoding='utf-8')[:100000]

X, y = np.array(word_emb.translate_and_shift(text_data))
X = np.array([X])
y = np.array([y])

# picking out a number of words (24000) that can be divided by batchsize=16
X = X[:, :24000, :]
y = y[:, :24000, :]
print('Shape of X after picking first 24000 words', X.shape)
print('Shape of y after picking first 24000 words', y.shape)

vocab, inverse_vocab = text_proc.create_vocabulary(X)
y = text_proc.create_labels(X, inverse_vocab)
print('Shape of X after onehot-encoding of y:', X.shape)
print('Shape of y after onehot-encoding of y:', y.shape)
# word_ix = np.argwhere(y[0, -1]).flatten()[0]
# word = vocab[word_ix]
# closest = word_emb.find_closest(word.flatten(), 1)
# print(closest)
# exit()

X = X.reshape(1, -1, num_batches, X.shape[-1])
y = y.reshape(1, -1, num_batches, y.shape[-1])
print('Shape of X after batching:', X.shape)
print('Shape of y after batching:', y.shape)


train = True
train = True
if train:
    rnn_batch = RNN_parallel(
        hidden_activation='Tanh()',
        output_activation='Softmax()',
        loss_function='ce()',
        optimiser=optimiser,
        clip_threshold=1,
        learning_rate=learning_rate,
        name='test_tf_dataset_m2_rng'
        )

    hidden_state_batch = rnn_batch.fit(
        X,
        y,
        epo,
        num_hidden_nodes=hidden_nodes,
        num_backsteps=num_backsteps,
        num_forwardsteps=num_forwardsteps,
        vocab=vocab,
        inverse_vocab=inverse_vocab,
        # gradcheck_at=3,
    )

rnn_batch = load_model('saved_models/test_tf_dataset_m2_rng')
X_seed = np.array([word_emb.get_embeddings('ROMEO')])
predict = rnn_batch.predict(X_seed.reshape(-1, 1, X_seed.shape[-1]), time_steps_to_generate=10)
for emb in predict:
    print(word_emb.find_closest(emb.flatten(), 1))

plt.plot(rnn_batch.get_stats()['loss'], label='batch train')
plt.legend()
plt.show()
