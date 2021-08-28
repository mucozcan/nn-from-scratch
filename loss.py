import numpy as np

from utils import sigmoid_prime

class MSELoss():

    @staticmethod
    def fn(y_hat, y):

        return 0.5*np.linalg.norm(y_hat - y)**2

    @staticmethod
    def delta(z, y_hat, y):
        """Returns the error delta from the output layer."""

        return (y_hat, y)  * sigmoid_prime(z)



class BCELoss():
    """Binary cross-entropy loss."""
    @staticmethod
    def fn(y_hat, y):

        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 -
                                                                          y_hat)))
    @staticmethod
    def delta(z, y_hat, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (y_hat - y)









