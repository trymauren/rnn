from rnn_not_nonsense import RNN_NOT_NONSENSE
from utils.text_processing import WORD_EMBEDDING
import numpy as np

rnn = RNN_NOT_NONSENSE(train=False)
word_embeddings = WORD_EMBEDDING()
#X_data = word_embedding.get_embeddings(word_embeddings,
#"There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.")

X_data = np.array([np.sin(np.linspace(0,4*np.pi,100))]).T
print(X_data)
rnn.fit(X=X_data,
        y = X_data,
        epochs = 1000,
        num_hidden_states = 5,
        num_hidden_nodes = 5
        )
#pred_x = word_embedding.get_embeddings(word_embeddings,"There was a big black cat in the house")
pred_x = np.array([np.sin(np.linspace(0,4*np.pi,100))]).T
print(rnn.predict(pred_x))