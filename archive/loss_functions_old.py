import sys
import git
import numpy as np
from collections.abc import Callable
import jax
from jax import grad
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # for gradient checking precision

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

LOG_CONST = 1e-15  # why this number? Many use it


class LossFunction(Callable):

    def __call__(self, y, y_pred, nograd=False):
        return self.eval(y, y_pred, nograd=nograd)


class Mean_Square_Loss(LossFunction):

    # def __init__(self):
    #     super().__init__()
    #     self.loss = None
    #     self.y_pred = None
    #     self.y_true = None

    # def eval(self, y_true, y_pred, nograd=False):
    #     if not nograd:
    #         self.y_pred = y_pred
    #         self.y_true = y_true
    #     loss = np.square(np.subtract(y_true, y_pred)).mean()
    #     print(loss)
    #     return loss

    # def grad(self):
    #     grad = (2
    #             * np.subtract(self.y_pred, self.y_true)
    #             / len(self.y_pred))
    #     self.loss = None
    #     if grad.ndim == self.y_pred.ndim:
    #         grad = np.expand_dims(grad, axis=grad.ndim)
    #     return grad

    def eval(self, y_true, y_pred, nograd=False):
        y_true = jnp.array(y_true, dtype=jnp.float64)
        y_pred = jnp.array(y_pred, dtype=jnp.float64)
        if not nograd:
            self.y_pred = y_pred
            self.y_true = y_true
        loss = self.jax_loss(y_true, y_pred)
        return np.asarray(loss, dtype=np.float64)

    def grad(self):
        grad = (
                # 1
                2
                * np.subtract(self.y_pred, self.y_true)
                / len(self.y_pred)
                )
        return grad

    def jax_loss(self, y_true, y_pred):
        return jnp.square(jnp.subtract(y_true, y_pred)).mean(dtype=jnp.float64)

    def grad_2(self):

        grad = jax.grad(self.jax_loss, argnums=0)
        return grad(self.y_true, self.y_pred)


class Classification_Logloss(LossFunction):

    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
        self.probabilities = None

    def eval(self, y_true, y_pred, nograd):
        y_pred += LOG_CONST  # to avoid log(0) calculations
        # probabilities = np.exp(y_pred)/np.sum(np.exp(y_pred))
        probabilities = y_pred
        if not nograd:
            self.y_pred = y_pred
            self.y_true = y_true
            self.probabilities = probabilities

        return -np.sum(y_true*np.log(probabilities), axis=0)

    def grad(self):
        probabilities = np.copy(self.probabilities)
        # See deep learning book, 10.18 for
        # explanation of the following line.
        grad = probabilities - self.y_true
        if grad.ndim == self.y_true.ndim:
            grad = np.expand_dims(grad, axis=grad.ndim)
        return grad