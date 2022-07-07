"""
logging by Tensorboard
https://www.tensorflow.org/tensorboard?hl=ko

Reference for official pytorch tensorboard document
https://pytorch.org/docs/stable/tensorboard.html

"""
import os

import torch
from torch.utils.tensorboard import SummaryWriter


def get_writer(root_log, experiment):
    """
    return pytorch SummaryWriter (tensorboard)
    logging on logdir/experiment

    Input
    - root_log (str): root directory path for logger
    - experiment (str): experiment tag

    Output
    - writer (SummaryWriter): tensorboard writter for logging
    """

    log_dir = os.path.join(root_log, experiment)

    writer = SummaryWriter(log_dir=log_dir)

    return writer


def log_round_accuracy(writer, round, accuracy, suffix=''):
    """
    Logging round-accuracy at writer
    X-axis: round
    Y-axis: accuracy

    Input
    - writer (SummaryWriter): tensorboard writter for logging (ref. get_writer())
    - round (int): federated learning round
    - accuarcy (int): accuarcy at `round`
    - suffix (str, optional): identify tag

    Output
    - None
    """
    if not isinstance(suffix, str):
        suffix = str(suffix)

    tag = 'Accuracy' + suffix

    writer.add_scalar(tag, accuracy, round)


def log_round_loss(writer, round, loss, suffix=''):
    """
    Logging round-loss at writer
    X-axis: round
    Y-axis: loss

    Input
    - writer (SummaryWriter): tensorboard writter for logging (ref. get_writer())
    - round (int): federated learning round
    - loss (int): loss at `round`
    - suffix (str, optional): identify tag

    Output
    - None
    """
    if not isinstance(suffix, str):
        suffix = str(suffix)

    tag = 'Loss' + suffix

    writer.add_scalar(tag, loss, round)