import os, copy

from tqdm import tqdm
import torch

from datasets.getdata import get_traindata, get_evaldata
from FLframework.clients import Client
from clientselection.random_selection import random_selection

class Server:
    def __init__(self, model, args):
        self._model_server = model
        self.args = args
        self.possible_clients = self._init_clients()

    def _get_possible_clients(self):
        idx_list = []

        for client in self.possible_clients:
            idx_list.append(client)

        return idx_list

    def _init_clients(self, num_clients, model, args):
        """
        Initialize clients
        """
        clients = []
        num_gpus = torch.cuda.device_count()

        train_data = get_traindata(args)
        eval_data = get_evaldata(args)

        for client_id in range(num_clients):
            device_id = client_id % num_gpus # may be... distribute the same number of client for each gpu
            client = Client(client_id, train_data, eval_data, model, device_id, args)
            clients.append(client)

        return clients

    def _distribute_server(self):
        """
        Distribute global server model to clients.
        direct copy of global model? 
        https://discuss.pytorch.org/t/can-i-deepcopy-a-model/52192
        """
        for client in self.possible_clients:
            client.update_client(copy.deepcopy(self._model_server))

    def _train_client(self):
        """
        train each client
        """
        pbar = tqdm(self.possible_clients, desc='Train clients') # progress bar

        for client in self.possible_clients:
            loss = client.train_client()
            pbar.set_postfix({'loss': loss})

    def _select_client(self):
        """
        select client
        """
        selected_clients = []

        if self.args.method == 'Random':
            selected_clients = random_selection(self.possible_clients)
        elif self.args.method == 'MaxEntropy':
            pass
        elif self.args.method == 'MinEntropy':
            pass
        elif self.args.method == 'MaxLoss':
            pass
        elif self.args.method == 'MinLoss':
            pass

        assert len(selected_clients) > 0

        return selected_clients

    def _aggregate_client(self, clients):
        """
        aggreate information (score, loss or entropy etc.) from selected clients 
        *FIXIT* temporaly FedAVG, encapsulate each algorithm and... apply here

        input
        - clients (list of Client): selected client list

        output
        None
        """
        pbar = tqdm(clients, desc='Aggregate clients')

        for client in pbar:
            pass

    def _update_server(self):
        pass

    def _eval_server(self):
        total_loss = 0.0
        correct, total = 0, 0

        for client in self.possible_clients:
            client_loss, correction = client.eval_client()

            correct += correction[0]
            total += correction[1]
        
        accuracy = correct * 1.0 / total

    def run_round(self):
        """
        run one rounds
        """
        self._distribute_server()
        self._train_client()
        selected_clients = self._select_client()
        self._aggregate_client(selected_clients)
        self._update_server()
        self._eval_server()

    def save_model(self):
        pass