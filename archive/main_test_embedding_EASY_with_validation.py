import sys
import git
import resource
import numpy as np
from numpy.linalg import norm
# from rnn.rnn import RNN
from rnn.rnn_batch_new import RNN
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model
from utils.loss_functions import Mean_Square_Loss
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


# import tensorflow as tf
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
# text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it

word_emb = WORD_EMBEDDING()

#text_data = text_proc.read_file(path_to_file)
text_data = text_proc.read_file("data/three_little_pigs.txt")
X, y = np.array(word_emb.translate_and_shift(text_data))
# print('X:', X.shape)
# print('y:', y.shape)
# text_data = text_data.split('.')
X = np.array([X])
y = np.array([y])
X_val = X[:, :100, :]  # pick first 100 samples as validation set
y_val = y[:, :100, :]  # pick first 100 samples as validation set

vocab, inverse_vocab = text_proc.create_vocabulary(X)
y = text_proc.create_labels(X, inverse_vocab)
X = X.reshape(1, -1, 2, X.shape[-1])
y = y.reshape(1, -1, 2, y.shape[-1])
X_val = X_val.reshape(1, -1, 2, X_val.shape[-1])
y_val = y_val.reshape(1, -1, 2, y_val.shape[-1])
print('X:', X.shape)
print('y:', y.shape)


def model_performance_embeddings(rnn, X, num_tests, test_length = 20, seed_length = 10):
    """
    Parameters:
    ------------------------------------------------------
    rnn:
        - rnn model to measure performance on.
    X:
        - dataset to make predictions on.
    num_tests:
        - number of predictions to base performance measure upon
    test_length:
        - length of each prediction
    seed_length:
        - length of input/seed to base a prediction upon
    
    Returns:
    -------------------------------------------------------
    cosine similarity between prediction and actual text:
        - float
    """
    similarities = []
    mse = Mean_Square_Loss()

    for i in range(num_tests):
        start_idx = np.random.randint(0,len(X[0][0])-seed_length)

        X_seed = X[0][0][start_idx:start_idx+seed_length][:]
        y_true = X[0][0][start_idx + seed_length : start_idx + seed_length + test_length][:]

        predict = rnn.predict(X_seed, test_length)
        cos_similarity = []
        for pred_emb, true_emb in zip(predict,y_true):
            if np.sum(pred_emb) != 0 and np.sum(true_emb) != 0:
                #print(np.dot(pred_emb,true_emb)/(norm(pred_emb)*norm(true_emb)))
                cos_similarity.append(np.dot(pred_emb,true_emb)/(norm(pred_emb)*norm(true_emb)))
        #print(cos_similarity)
        similarities.append(np.mean(cos_similarity))
    return np.round(np.mean(similarities),3)


train = True

infer = True
if train:

    epo = 10
    hidden_nodes = 300
    # learning_rates = [0.001, 0.003, 0.005, 0.01]

    rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Softmax()',
        loss_function='Classification_Logloss()',
        optimiser='AdaGrad()',
        clip_threshold=np.inf,
        name='new',
        learning_rate=0.001,
        )

    hidden_state = rnn.fit(
        X,
        y,
        epo,
        num_hidden_nodes=hidden_nodes,
        num_forwardsteps=30,
        num_backsteps=30,
        vocab=vocab,
        inverse_vocab=inverse_vocab,
        X_val=X_val,
        y_val=y_val,
        num_epochs_no_update=10
        )
    rnn.plot_loss(plt, show=True)

if infer:

    X_seed = np.array([word_emb.get_embeddings("What should")])
    rnn = load_model('saved_models/new')
    rnn.plot_loss(plt, show=True)
    predict = rnn.predict(X_seed.reshape(-1, 1, X_seed.shape[-1]), time_steps_to_generate=10)
    for emb in predict:
        print(word_emb.find_closest(emb, 1))
    # print(model_performance_embeddings(rnn, X, 10))

bytes_usage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
gb_usage_peak = round(bytes_usage_peak/1000000000, 3)
print('Memory consumption (peak):')
print(gb_usage_peak, 'GB')



# for learning_rate_curr in learning_rates:
#    fig, ax = plt.subplots()
#    print(f'learning rate: {learning_rate_curr}')
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='AdaGrad()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr)
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=False)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))
#
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='SGD()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr
#        )
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=False)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))
#
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='SGD_momentum()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr,
#        momentum_rate=0.9)
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=False)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))
#
#    rnn = RNN(
#        hidden_activation='Tanh()',
#        output_activation='Identity()',
#        loss_function='mse()',
#        optimiser='RMSProp()',
#        regression=True,
#        threshold=1,
#        )
#
#    whole_sequence_output, hidden_state = rnn.fit(
#        X, y, epo,
#        num_hidden_nodes=hidden_nodes, return_sequences=True,
#        independent_samples=True, learning_rate=learning_rate_curr,
#        decay_rate=0.001)
#
#    rnn.plot_loss(plt, figax=(fig, ax), show=True)
#
#    predict = rnn.predict(X_seed)
#    for emb in predict:
#        print(word_emb.find_closest(emb,1))

# plt.plot(rnn.get_stats()['loss'])
# plt.show()
# x_seed = X[0][0]
# print(word_emb.find_closest(x_seed, number=1))
# ret = rnn.predict(x_seed, hidden_state, 10)

# for emb in ret:
#     word = word_emb.find_closest(emb, number=1)
#     print(word)
