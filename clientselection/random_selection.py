import numpy as np

def random_selection(client_candidate, num_select):
    """
    Random client selection method

    Input
    - client_candidate (list of client index): candidate client index
    - num_select (int): selection number

    Output
    - select_client (list of client idx): selected client list with size `num_select`
    """
    select_client = np.random.choice(client_candidate, num_select, replace=False)
    return select_client