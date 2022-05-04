import os
from typing import Tuple

def make_folder(path:str) -> Tuple[str, str]:
    """ Create directory to save results and mode.

    Parameters
    ----------
    path : str
        main path where the directory should be located

    Return:
    ---------
    folder_model : str
        directory for the model to be saved at
    folder_result : str
        directory where the results are stored

    """
    if not os.path.exists(path):
        os.mkdir(path)
    folder_model = os.path.join(path, 'soc_model')
    if not os.path.exists(folder_model):
        os.mkdir(folder_model)
    folder_result = os.path.join(path, 'soc_result')
    if not os.path.exists(folder_result):
        os.mkdir(folder_result)
    return folder_model, folder_result
