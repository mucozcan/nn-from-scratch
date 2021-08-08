import numpy as np


class Network(object):
    def __init__(self, layers):
        """Base class of an artificial neural network.

        Args:
            layers (List): number of neurons in each layer. 
            Ex.: [2, 3, 1]. 2 neurons in input layer, 3 neuron in hidden layer and 1 neuron in output layer.

        """

        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y,1) for layer in layers[1:]] # bias initalization. We skipped input layer.
        self.weights = [np.random.randn(y,x) for x, y in zip(layers[:-1], layers[1:])] # weights initialization

    def feedforward(self, x):
        """Return the output of network

        Args:
            x (np.array): input that feed to the network.
        """

        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)

        return x


def sigmoid(z):
    """Sigmoid Function

    Args:
        z (np.array): w.x(multiplication of all inputs and their weights) + b(bias of the neuron to activate) 

    Returns:
        np.array: output of sigmoid
    """
    return 1.0/(1.0 + np.exp(-z))