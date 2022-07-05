import os, copy
import torch

from datasets.getdata import get_traindata, get_evaldata
from FLframework.clients import Client
from clientselection.random_selection import random_selection

class Server:
    def __init__(self, model, args):
        self._model_server = model
        self.args = args
        self.possible_clients = self._init_clients()
        self.selected_clients = []

    def _get_possible_clients(self):
        idx_list = []

        for client in self.possible_clients:
            idx_list.append(client)

        return idx_list

    def _init_clients(self, num_clients, model, args):
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
        direct copy of global model? 
        https://discuss.pytorch.org/t/can-i-deepcopy-a-model/52192
        """
        for client in self.possible_clients:
            client.update_client(copy.deepcopy(self._model_server))

    def _train_client(self):
        """
        train each client
        """
        for client in self.possible_clients:
            client.train_client()

    def _select_client(self):
        """
        select client
        """
        self.selected_clients = []

        if self.args.method == 'Random':
            self.selected_clients = random_selection(self._get_possible_clients())

        assert len(self.selected_clients) > 0

    def _aggregate_client(self, clients):
        """
        aggreate information (score, loss or entropy etc.) from selected clients
        input
        - clients (list of Client): selected client list
        output
        None
        """
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