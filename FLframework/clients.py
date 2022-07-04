import os

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

def get_optimizer(model, optimizer_str, lr):
    optimizer = None

    if optimizer_str == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_str == 'Adam':
        optimizer = Adam(model.parameters(), )

    assert optimizer != None
    return optimizer


def get_lossf(loss_str):
    lossf = None

    if loss_str == 'CE':
        lossf = nn.CrossEntropyLoss()
    elif loss_str == 'MSE':
        lossf = nn.MSELoss()
    elif loss_str == 'L1':
        lossf = nn.L1Loss()

    assert lossf != None
    return lossf


class Clients:
    def __init__(self,
        client_id,
        train_data,
        eval_data,
        model,
        args):
        """
        initialize clients
        Input
        - train_data
        Output
        - None
        """
        # save arguments
        self._id = client_id
        self._model = model
        self._train_data = train_data
        self._eval_data = eval_data
        self._lossf = get_lossf(args.loss_function_client)

        # get information from args
        self.lr = args.lr_client
        self.optimizer_str = args.optimizer

        self.batch_size = args.batch_size_client
        self.epochs = args.num_epoch_client
        self.pin_memory = args.pin_memory_client
        self.num_workers = args.num_workers_client
        
        if args.device == 'cuda':
            self.device = 'cuda:' + args.gpu_id
        else:
            self.device = 'cpu'

        self.loss = 0.0

    def _train_epoch(self, dataloader, optimizer):
        running_loss = 0.0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            preds = self._model(X)
            loss = self._lossf(preds, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        self.loss = running_loss

    def _eval_epoch(self, dataloader):
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                preds = self._model(X)
                loss = self._lossf(preds, y)

                total_loss += loss.item()
                correct += (preds.argmax(1) == y).type(torch.float).sum().item()

        return total_loss, correct

    def _get_loss(self):
        return self.loss

    def _get_entropy(self):
        pass

    def get_client_id(self):
        return self.client_id

    def update_client(self, model):
        self._model = model

    def train_client(self):
        dataloader = DataLoader(self._train_data, batch_size=self.batch_size, 
            shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)
        optimizer = get_optimizer(self._model, self.optimizer_str, self.lr)

        for epoch in self.epochs:
            self._train_epoch(dataloader, optimizer)

    def eval_client(self):
        dataloader = DataLoader(self._eval_data, batch_size=self.batch_size, 
            shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)
        
        total_loss, correct = self._eval_epoch(dataloader)

        return total_loss, (correct, len(self._eval_data))

    def get_size_info(self):
        data_sz = len(self._train_data)
        return data_sz