import torch

from utils.argparser import arg_parser

def run_round():
    """
    run one round
    """
    pass


def main():
    args = arg_parser()

    # make save directory

    # initialize clients

    # simulate round
    num_rounds = args.num_round
    for round in range(num_rounds):
        run_round()

    # save server model

if __name__ == '__main__':
    main()