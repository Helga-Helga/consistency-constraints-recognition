from numpy.random import laplace
from numpy import clip


def add_laplacian_noise(image, loc=1.0, scale=10.0):
    """Add Laplacian noise to image.
    It has probability density function
    f(x; mu, lambda) = 1 / (2 * lambda) * exp(-|x - mu| / lambda)

    Parameters
    ----------
    image : numpy 2D array
        Input image
    loc : float
        The position, mu, of the distribution peak
    scale: float
        lambda, the exponential decay

    Returns
    -------
    numpy 2D array
        Image with Laplacian noise
    """
    noise = laplace(loc, scale, image.shape)
    return clip(image + noise, 0, 255)
