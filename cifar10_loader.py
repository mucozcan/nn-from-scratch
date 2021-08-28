import pickle
import os
import numpy as np

from utils import one_hot_encode

class CIFAR10():
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir,"train")
        self.test_dir = os.path.join(root_dir,"test")

        self.train_data = None
        self.train_labels = None

        self.test_data = None
        self.test_labels = None
        self.batch_iterator = 0


    def load_data(self, flatten=False):
        train = []
        for file in os.listdir(self.train_dir):
            with open(os.path.join(self.train_dir, file), 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                train.append(batch)
        
        self.train_data = np.vstack([batch[b"data"] for batch in train])
        train_len = len(self.train_data)

        self.train_data = self.train_data.reshape(train_len, 3, 32, 32).transpose(0,2,3,1) / 255
        self.train_labels = one_hot_encode(np.hstack([
            batch[b"labels"] for batch in train
        ]), 10)
        print("Train data is loaded.")
        test = []
        for file in os.listdir(self.test_dir):
            with open(os.path.join(self.test_dir, file), 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                test.append(batch)
        
        self.test_data = np.vstack(batch[b"data"] for batch in test)
        test_len = len(self.test_data)

        self.test_data = self.test_data.reshape(test_len, 3, 32, 32).transpose(0,2,3,1) / 255
        self.test_labels = one_hot_encode(np.hstack([
            batch[b"labels"] for batch in test
        ]), 10)

        print("Test data is loaded.")

        if flatten:
            self.train_data = self.train_data.flatten().reshape(train_len, 3072)
            self.test_data = self.test_data.flatten().reshape(test_len, 3072, 1)

            self.test_data = [self.test_data, self.test_labels]
            
    def next_batch(self, batch_size):
        x = self.train_data[self.batch_iterator : self.batch_iterator + batch_size]
        y = self.train_labels[self.batch_iterator : self.batch_iterator + batch_size]
        self.batch_iterator = (self.batch_iterator + batch_size) % len(self.train_data)
        return x, y



if __name__ == "__main__":
    cifar_dataset = CIFAR10("./data/cifar-10-batches-py/")
    cifar_dataset.load_data(flatten=True)

    x, y = cifar_dataset.next_batch(2)
    print(x.shape)
    print(y.shape)
