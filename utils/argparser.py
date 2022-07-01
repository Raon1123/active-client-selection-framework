import argparse

from utils.constants import *

def arg_parser():
    parser = argparse.ArgumentParser()

    # device information
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
        choices=OPTIMIZER, help='local update optimizer')
    parser.add_argument('--lr_local', type=float, default=0.1, help='local updating learning rate')
    parser.add_argument('--batch_size_local', type=int, default=64)
    parser.add_argument('--num_epoch_local', type=int, default=1)

    # logger information
    parser.add_argument('--logdir', type=str, default='logdir', help='root directory of logging')

    args = parser.parse_args()
    return args