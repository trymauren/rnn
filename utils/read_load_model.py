import sys
import git
import pickle


def save_model(estimator, path):
    """
    Given a trained (or not trained) model, together with the path
    and filename, this function will dump it using pickle
    See: https://docs.python.org/3/library/pickle.html

    TODO: Use something that is safe instead..?
    NOTE: But don't we like a life one the edge?
    """
    with open(path, 'wb') as filepointer:
        pickle.dump(
                    estimator,
                    filepointer,
                    protocol=pickle.HIGHEST_PROTOCOL
                    )


def load_model(path):
    """
    Given a file path (that includes the filename), this function
    will load it from a file that has been written using pickle.
    See: https://docs.python.org/3/library/pickle.html

    TODO: Use something that is safe instead..?
    NOTE: But don't we like a life one the edge?
    """
    with open(path, 'rb') as filepointer:
        estimator = pickle.load(filepointer)

    return estimator
