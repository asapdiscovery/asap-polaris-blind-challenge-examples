import numpy as np


def bootstrapping_sampler(size: int, n_samples: int) -> np.ndarray:
    """
    Generate bootstrap samples for a given size and number of samples.

    Parameters
    ----------
    size : int
        The size of the data.
    n_samples : int
        The number of samples to generate.

    Returns
    -------
    np.ndarray
        Returns a numpy array of the bootstrap samples.
    """
    rng = np.random.default_rng(0)
    return rng.choice(size, size=(n_samples, size), replace=True)
