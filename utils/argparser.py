import argparse

from utils.constants import *

def arg_parser():
    parser = argparse.ArgumentParser()

    # device information
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

    # dataset information
    parser.add_argument('--dataset', type=str, choices=DATASETS, help='datasets string')
    parser.add_argument('--data_root', type=str)

    # client selection
    parser.add_argument('--method', type=str, default='Random',
        choices=CLIENTSELECTION, help='active client seleciton methods')

    # define global model
    parser.add_argument('--model', type=str, default='CNN', 
        choices=MODELS, help='global model')
    parser.add_argument('--update_algo', type=str, default='FedAVG', 
        help='Federated algorithm for update global model')
    parser.add_argument('--lr_global', type=float, default=0.001, help='global model learning rate')
    parser.add_argument('--num_round', type=int, default=2000)

    # client setting
    parser.add_argument('--optimizer', type=str, default='SGD',
        choices=OPTIMIZER, help='client update optimizer')
    parser.add_argument('--lr_client', type=float, default=0.1, help='client updating learning rate')
    parser.add_argument('--batch_size_client', type=int, default=64)
    parser.add_argument('--num_epoch_client', type=int, default=1)
    parser.add_argument('--pin_memory_client', type=bool, default=True, help='pin memory of dataloader in client')
    parser.add_argument('--num_workers_client', type=int, default=1, help='number of cpu workers at client dataloader')
    parser.add_argument('--loss_funciton_client', type=str, default='CE', 
        choices=['CE', 'MSE', 'L1'], help='loss function for client')

    # logger information
    parser.add_argument('--logdir', type=str, default='./logdir', help='root directory of logging')

    args = parser.parse_args()
    return args