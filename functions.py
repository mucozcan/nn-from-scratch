import numpy as np

def sigmoid(z):
    """Sigmoid Function

    Args:
        z (np.array): w.x(multiplication of all inputs and their weights) + b(bias of the neuron to activate) 

    Returns:
        np.array: output of sigmoid
    """
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))