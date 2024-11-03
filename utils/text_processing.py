import numpy as np
import spacy
import warnings
import sys
import git

def read_file(path: str) -> np.ndarray:
    """
    Parses the given file into a numpy array containing one long string
    up to 100 000 charcters long, the max length is set to be within
    Spacy's max tokenization length.
    Parameters:
    --------------------------------
    path : str
    - ABSOLUTE path to text file to be parsed as a string

    Returns:
    --------------------------------
    numpy.ndarray
    - the text as a single long string as an entry in the array
    """
    text = ""
    windowed_text = []
    with open(path, 'r') as f:
        text = f.read()
        text = text.replace("\n", "")
        len_chars = sum(len(word) for word in text.strip().split())
        if len_chars > 100000:
            text = text[:100000]
            warnings.warn("Length of passed text exceeded limit of 100 000\
                          charcters. Only first 100 000 charcters kept and\
                          parsed")
    return text

def create_vocabulary(word_embeddings: np.ndarray) -> dict:
    unique_embeddings = np.unique(word_embeddings[0], axis=0)
    vocabulary = dict(zip(range(len(unique_embeddings)), unique_embeddings))
    inverse_vocabulary = dict(zip(tuple(map(tuple, unique_embeddings)),
                                  range(len(unique_embeddings))))
    return vocabulary, inverse_vocabulary

def create_labels(X, inverse_vocabulary) -> np.ndarray:
    y = []
    for embedding in X[0]:
        y.append([inverse_vocabulary[tuple(embedding)]])
    y = np.array(y)
    one_hot_y = np.zeros((len(y), y.max()+1))
    for value, i in zip(y, range(len(y))):
        one_hot_y[i, value] = 1
    return np.array([one_hot_y])

def create_onehot(chars: str, char_to_ix: dict):
    """
    Converts a string of characters into an array containing
    onehot-encoded arrays

    Parameters:
    ---------------------------
        chars : 
        - string of characters to be converted into one-hot vectors

    Returns:
    ---------------------------
        X : 
        - np.ndarray, shape: (str_to_convert, 1, len(char_to_ix))
    """

    X = np.zeros((len(chars), 1, len(char_to_ix)))

    for t in range(len(chars)):
        inp = (np.zeros((len(char_to_ix), 1)))
        inp[chars[t]] = 1
        X[t] = inp.T

    return X


def onehot_to_ix(onehot):
    """
    Returns the index of the 1 from a onehot-array
    """
    return onehot.argmax()

class WORD_EMBEDDING():
    """
    Class for initializing a word embedding dataset and converting text
    to and from word embeddings.
    """

    def __init__(self) -> None:
        """
        Init function for loading the embedding dataset
        """
        self.nlp = spacy.load("en_core_web_lg")
        import en_core_web_lg
        self.nlp = en_core_web_lg.load()

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Translates the given string into word embeddings/vectors by
        tokenizing the string and then translating the tokens into word
        embeddings

        Parameters:
        --------------------------
        text : str
        - as string of the text/word that is to be tokenized and
          translated into embedding(s)

        Returns:
        --------------------------
        np.ndarray:
        - numpy array of word embeddings of a given size depending on 
          the embedding dataset
        """
        embeddings = []
        doc = self.nlp(text)
        for token in doc:
            embeddings.append(token.vector)
        return np.array(embeddings)

    def find_closest(self, embedding: np.ndarray, number: int) -> np.ndarray:
        """
        Finds and returns the n nearest tokens(as characters, not
        embeddings) to the passed embedding in the vector space of the 
        word embedding dataset using euclidean distance

        Parameters:
        ---------------------------
        embedding : np.ndarray
        - A 1d word embedding vector of length matching the embedding 
          dataset

        number : int
        - parameter specifiying to find the n closest tokens to the 
          passed word embedding

        Returns:
        ---------------------------
        np.ndarray:
        - numpy array containing the n closest tokens the the given word
          embedding
        """
        most_similar = self.nlp.vocab.vectors.most_similar(
                                                np.array([embedding]),
                                                n=number)
        keys = most_similar[0][0]
        nearest_words = []
        for key in keys:
            nearest_words.append(self.nlp.vocab.strings[key])
        return np.array(nearest_words)

    def translate_and_shift(self, data: str):
        """
        Translate text data from string into 2 separate embedding sets,
        the first ranging from 0->(N-1), and the second ranging
        from 1 -> N

        Parameters:
        ---------------------------
            data : str
            - text data as a single string of up to 100 000 characters
              long

        Returns:
        ---------------------------
            x : np.ndarray
            - text data translated into embeddings, covers 0->(len(X)-1)
              of X

            y : np.ndarray
            - text data translated into embeddings, covers 1->len(X) of 
              X, meant for validation of x
        """
        word_embs = self.get_embeddings(data)
        x = word_embs[0:-1]
        y = word_embs[1:]
        return x, y


