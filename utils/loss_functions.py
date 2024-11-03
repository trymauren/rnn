import sys
import git
import numpy as np
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

LOG_CONST = 1e-15  # Used to avoid log(0)


class LossFunction():
    """ Wrapper class for loss functions """

    def __call__(self, y_true, y_pred, nograd=False):
        """
        Calls the eval method to compute the loss.

        Parameters:
        -------------------------------
        y_true: np.ndarray
            - True values

        y_pred: np.ndarray
            - Estimated values

        nograd: bool
            - If True, do not store values for gradient computation.

        Returns:
        -------------------------------
        computed loss: float

        """
        return self.eval(y_true, y_pred, nograd=nograd)


class Mean_Square_Loss(LossFunction):

    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None

    def eval(self, y_true, y_pred, nograd=False):
        """
        Returns the mean squared loss of (y_true, y_pred)

        """

        if not nograd:
            self.y_pred = y_pred.copy()
            self.y_true = y_true.copy()

        loss = np.square(np.subtract(y_true, y_pred)).mean(dtype=y_pred.dtype)
        return loss

    def grad(self):
        """
        Computes the gradient of the mean squared loss w/ respect to
        y_pred

        """

        grad = (
                2
                * np.subtract(self.y_pred, self.y_true)
                / self.y_pred.size
                )
        return grad


class Classification_Logloss(LossFunction):

    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
        self.probabilities = None

    def eval(self, y_true, y_pred, nograd):
        """
        Returns the logarithmic loss of (y_true, y_pred)

        """

        probabilities = y_pred + LOG_CONST
        if not nograd:
            self.y_pred = np.copy(y_pred)
            self.y_true = np.copy(y_true)
            self.probabilities = np.copy(probabilities)

        return -np.mean(np.sum(np.log(probabilities) * y_true))

    def grad(self):
        """
        Computes the gradient of the logarithmic loss w/ respect to
        y_pred

        """

        # See deep learning book, 10.18 for
        # explanation of the following line.
        grad = self.probabilities - self.y_true
        return grad
