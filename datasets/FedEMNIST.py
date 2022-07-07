import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

class FederatedEMNIST:
    def __init__(self,
        num_clients,
        alpha,
        train=True):
        """
        Initialize dataset initializer,
        For non-iid setting, using Dirichlet distribution

        Input
        - num_clients: total number of clients for using
        - alpha: alpha for Dirichlet distribution
        - train: for training data or not
        """
        self.num_classes = 62
        self.num_clients = num_clients
        self.alpha = alpha
        self.train = train

    def __len__(self):
        return self.num_clients

    def __getitem__(self, idx):
        pass

    def read_data(self, data_root):
        """
        Read and parse raw data

        Input
        - data_root (str): root directory of dataset
        """
        pass

    def generate_partition(self):
        partition = np.random.dirichlet([self.alpha] * self.num_classes, self.num_clients) # clients * classes

    