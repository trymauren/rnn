import sys
import git
import numpy as np
from rnn.rnn import RNN
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
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
fig, ax = plt.subplots()

seq_length = 20
word_emb = WORD_EMBEDDING()
# X = [word_emb.get_embeddings(str(s)) for s in text_proc.read_file(
#     'utils/embedding_test.txt',
#     seq_length)]

# print("X shape " + str(X.shape))
text = text_proc.read_file('utils/embedding_test_y.txt', seq_length)
for s in text:
    emb = word_emb.get_embeddings(str(s))
    print(len(emb))


print(X)
exit()

epo = 1000
hidden_nodes = 300
rnn = RNN(
    hidden_activation='Tanh()',
    output_activation='Identity()',
    loss_function='mse()',
    optimiser='AdaGrad()',
    regression=True,
    threshold=1,
    learning_rate=0.005,
    )

whole_sequence_output, hidden_state = rnn.fit(
    X, y, epo,
    num_hidden_nodes=hidden_nodes, return_sequences=True,
    independent_samples=True)

rnn.plot_loss(plt, figax=(fig, ax), show=False)
