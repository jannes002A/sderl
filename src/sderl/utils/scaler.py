import numpy as np
import sklearn.preprocessing

def get_scaler(env):
    """ get standard scaler object for scaling the state variable.
        Scaling the state variables makes the learning process of the NN easier.

    Parameters
    ----------
    env : object
        current used environment

    Returns
    -------
    scaler : object
        trained scaler on the input space
    """
    state_space_samples = np.linspace(env.min_position, env.max_position, 1000).reshape(-1, 1)  # returns shape =(1,1)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    return scaler
