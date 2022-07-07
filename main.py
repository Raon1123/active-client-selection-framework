import numpy as np

import torch
import random
import torch.backends.cudnn as cudnn

from utils.argparser import arg_parser

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(seed)
    

def run_round():
    """
    run one round
    """
    pass


def main():
    args = arg_parser()

    # fix seed
    fix_seed(args.random_seed)

    # make save directory

    # preprocessing

    # initialize clients

    # simulate round
    num_rounds = args.num_round
    for round in range(num_rounds):
        run_round()

    # save server model

if __name__ == '__main__':
    main()