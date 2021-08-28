import numpy as np
import random
import sys
import json

from utils import sigmoid, sigmoid_prime

class Network(object):
    def __init__(self, layers, loss):
        """Base class of an artificial neural network.

        Args:
            layers (List): number of neurons in each layer. 
            Ex.: [2, 3, 1]. 2 neurons in input layer, 3 neuron in hidden layer and 1 neuron in output layer.

        """

        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [(np.random.randn(y, x) / np.sqrt(x)) for x, y in zip(self.layers[:-1], self.layers[1:])]
        # self.param_initialize()
        self.loss = loss

    def param_initialize(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """

        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.rand(y, x) / np.sqrt(x) for x, y in
                        zip(self.layers[:-1], self.layers[1:])]



    def feedforward(self, x):
        """Return the output of network

        Args:
            x (np.array): input that feed to the network.
        """

        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w.transpose(), x) + b)

        return x

    def SGD(
        self,
        dataset,
        epochs,
        batch_size,
        lr,
        lmbda,
        eval_data=None,
        monitor_eval_loss=False,
        monitor_eval_accuracy=False,
        monitor_train_loss=False,
        monitor_train_accuracy=False
    ):
        """Stochastic Gradient Descent. 

        Args:
            train_data (tuple): (input, label)
            epochs (int): the number of epochs to train for 
            batch_size (int): size of mini-batches to use for sampling.
            lr (float): learning rate.
            test_data (list, optional): Data for evaluation. Defaults to None.
        """
        n_test = len(eval_data[0]) if eval_data is not None else None
        n_train = len(dataset.train_data)
        eval_loss, eval_accuracy = [], []
        train_loss, train_accuracy = [], []

        for epoch in range(epochs):

            for _ in range(batch_size):
                batch = dataset.next_batch(batch_size)
                self.update(batch, lr, lmbda, n_train)

            print(f"Trained {epoch} epochs", end=" -- ")

            if monitor_train_loss:
                loss = self.total_loss(batch, lmbda, dtype="train")
                train_loss.append(loss)
                print(f"Batch loss: {loss}",end=" -- ")

            if monitor_train_accuracy:
                accuracy = self.accuracy(batch, dtype="train")
                train_accuracy.append(accuracy)
                print(f"Batch accuracy: {accuracy}", end=" -- ")

            if monitor_eval_loss:
                loss = self.total_loss(eval_data, lmbda, dtype="eval")
                eval_loss.append(loss)
                print(f"Evaluation loss: {loss}", end=" -- ")

            if monitor_train_accuracy:
                accuracy = self.accuracy(eval_data, dtype="eval")
                eval_accuracy.append(accuracy)
                print(f"Evaluation accuracy: {accuracy}", end=" -- ")

            print("\n")

            return eval_loss, train_loss, eval_accuracy, train_accuracy


    def update(self, batch, lr, lmbda, n_train):
        """ Update the network's parameters(w,b) by applying gradient descent using
            backpropagation to single batch.

        Args:
            batch (list): [input, label]
            lr (float): learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        x, y = batch
        for input, label in zip(x, y):
            print(input.shape)
            input = np.expand_dims(input, axis=1)
            label = np.expand_dims(label, axis=1)
            delta_nabla_b, delta_nabla_w = self.backprop(input, label)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #http://neuralnetworksanddeeplearning.com/chap1.html Equations: 20 and 21
        self.weights = [
            ((1 - lr*(lmbda / n_train)) * w - (lr / len(batch)) * nw for w, nw in zip(self.weights, nabla_w))
        ]
        self.biases = [(b - lr/len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

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
            z = np.dot(w, activation) + b ## TODO Fix shape align issue
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.loss.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta # ??
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # ??

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())

        return (nabla_b, nabla_w)

    def accuracy(self, batch, dtype="eval"):

        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation."""

        if dtype == "train":

            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in batch]

            return sum(int(x == y) for (x, y) in results)

        results = [(np.argmax(self.feedforward(x)), y) for x, y in zip(batch[0],
                                                                      batch[1])]
        return sum(int(x == np.where(y == 1)) for (x, y) in results)

    def total_loss(self, batch, lmbda, dtype="eval"):

        loss = 0.0

        for x, y in batch:
            a = self.feedforward(x)
            loss += self.loss.loss_fn(a, y) / len(batch)

        loss += 0.5 * (lmbda/len(batch))*sum(np.linarg.norm(w)**2 for w in
                                             self.weights)

        return loss

    def load(filename):

        with open(filename, "r") as f:
            data = json.load(f)

        loss = getattr(sys.modules[__name__], data["loss"])
        model = Network(data["layers"], loss=loss)
        model.weights = [np.array(w) for w in data["weights"]]
        model.biases = [np.array(b) for b in data["biases"]]

        return model



