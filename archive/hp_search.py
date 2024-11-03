import numpy as np
import random
from tqdm import tqdm
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model
from rnn.rnn import RNN


def hp_search(text_data, num_optimisations, max_grad_clip = 1.0, max_lr = 0.1, max_epo = 10000,
              max_hidden_nodes = 1000, max_backsteps = 50):
    
    word_emb = WORD_EMBEDDING()

    X,y = np.array(word_emb.translate_and_shift(text_data))
    print(X.shape)
    X = np.array([X])
    y = np.array([y])


    best_loss = np.inf
    best_params = None

    print("-------------------------\n\
          Optimizing parameters, please wait\n\
          -------------------------\n")

    for i in range(num_optimisations):
        grad_clip_param = float(random.uniform(0.00001,max_grad_clip))
        lr_param = float(random.uniform(0.00001,max_lr))
        epo_param = random.randrange(10,max_epo)
        hidden_node_param = random.randrange(1,max_hidden_nodes)

        param_bounds = [max_grad_clip, max_lr, max_epo, max_hidden_nodes, max_backsteps]
        parameters = {'gradient_clip': grad_clip_param, 'learning_rate': lr_param,
                    'epochs': epo_param, 'hidden_nodes': hidden_node_param}
        rnn = RNN(
            hidden_activation='Tanh()',
            output_activation='Softmax()',
            loss_function='Classification_Logloss()',
            optimiser='AdaGrad()',
            clip_threshold=parameters['gradient_clip'],
            name='three_little_pigs_large_embedding',
            learning_rate=parameters['learning_rate'],
            )

        rnn.fit(
            X, y, parameters['epochs'],
            num_hidden_nodes=parameters['hidden_nodes'],
            independent_samples=True, num_backsteps=max_backsteps)
        
        curr_loss = np.argmin(rnn.stats['loss'])
        
        if curr_loss < best_loss:
            best_params = parameters
            best_loss = curr_loss

            for key, max_param in zip(parameters,param_bounds):
                if parameters[key]/max_param >= 0.95:
                    max_param = max_param * 1.05

    print("-------------------------\n\
          Finished optimizing parameters\n\
          -------------------------\n")

    return best_params, best_loss