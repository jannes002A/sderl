import os
from typing import Tuple

from sderl.utils.config import RES_DIR_PATH

def make_folder(algo: str) -> Tuple[str, str]:
    """ Creates directory to save results and models.

    Parameters
    ----------
    algo : str
        name of the chosen rl algorithm

    Return:
    ---------
    model_dir_path : str
        directory for the model to be saved at
    result_dir_path : str
        directory where the results are stored

    """
    # project results directory
    if not os.path.exists(RES_DIR_PATH):
        os.mkdir(RES_DIR_PATH)

    # directory for the model parameters of the chosen algorithm
    model_dir_path = os.path.join(RES_DIR_PATH, f'{algo}_model')
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)

    # directory for the results of the chosen algorithm
    result_dir_path = os.path.join(RES_DIR_PATH, f'{algo}_result')
    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)

    return model_dir_path, result_dir_path
