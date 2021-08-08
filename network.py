import numpy as np
import random


from functions import sigmoid, sigmoid_prime

class Network(object):
    def __init__(self, layers):
        """Base class of an artificial neural network.

        Args:
            layers (List): number of neurons in each layer. 
            Ex.: [2, 3, 1]. 2 neurons in input layer, 3 neuron in hidden layer and 1 neuron in output layer.

        """

        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(layer,1) for layer in layers[1:]] # bias initalization. We skipped input layer.
        self.weights = [np.random.randn(y,x) for x, y in zip(layers[:-1], layers[1:])] # weights initialization

    def feedforward(self, x):
        """Return the output of network

        Args:
            x (np.array): input that feed to the network.
        """

        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)

        return x

    def SGD(self, train_data, epochs, batch_size, lr, test_data=None):
        """Stochastic Gradient Descent. 

        Args:
            train_data (tuple): (input, label)
            epochs (int): the number of epochs to train for 
            batch_size (int): size of mini-batches to use for sampling.
            lr (float): learning rate.
            test_data (list, optional): Data for evaluation. Defaults to None.
        """
        n_test = len(test_data) if test_data else None
        n_train = len(train_data)

        for epoch in range(epochs):
            random.shuffle(train_data)
            batches = [
                train_data[k : k + batch_size] for k in range(0, n_train, batch_size)
            ]
            for batch in batches:
                self.update(batch, lr)
            
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch} complete")
    
    def update(self, batch, lr):
        """ Update the network's parameters(w,b) by applying gradient descent using
            backpropagation to single batch.

        Args:
            batch (tuple): (input, label)
            lr (float): learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #http://neuralnetworksanddeeplearning.com/chap1.html Equations: 20 and 21
        self.weights = [
            w - (lr/len(batch))*nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (lr/len(batch))*nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """Returns a tuple (nabla_b, nabla_w) representing the gradient for the cost function.

        Args:
            x ([type]): [description]
            y ([type]): [description]
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # http://neuralnetworksanddeeplearning.com/chap1.html Equation 9
        nabla_b[-1] = delta # ??
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # ??

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer+1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Returns the vector of partial derivatives of MSE loss. 

        Args:
            output_activations (list)
            y (int): ground truth
        """
        return (output_activations - y) # see  https://datascience.stackexchange.com/a/52159


