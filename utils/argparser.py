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

    # client setting
    parser.add_argument('--optimizer', type=str, default='SGD',
        choices=OPTIMIZER, help='local update optimizer')

    args = parser.parse_args()
    return args