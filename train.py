import numpy as np

from network import Network
from cifar10_loader import CIFAR10


dataset = CIFAR10("./data/cifar-10-batches-py/")
dataset.load_data(flatten=True)
model = Network([3072, 30, 10])
model.SGD(dataset, 30, 10, 3.0, dataset.test_data)