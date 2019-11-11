from numpy.random import (
    laplace,
    normal,
    random
)
from numpy import clip


def add_laplacian_noise(image, loc=0.0, scale=1.0):
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


def add_gaussian_noise(image, loc=0.0, scale=1.0):
    """Add Gaussian noise to image

    Parameters
    ----------
    image : numpy 2D array
        Input image
    loc : float
        Mean (center) of the distribution
    scale: float
        Standard deviation (spread or width) of the distribution

    Returns
    -------
    numpy 2D array
        Image with Gaussian noise
    """
    noise = normal(loc, scale, image.shape)
    return clip(image + noise, 0, 255)


def add_salt_and_pepper_noise(image, probability):
    """Add salt and pepper noise to image

    Parameters
    ----------
    image : numpy 2D array
        Input image
    probability : float
        Probability of the noise

    Returns
    -------
    numpy 2D array
        Image with salt and pepper noise
    """
    noised_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            random_uniform = random()
            if random_uniform < probability:
                noised_image[i][j] = 0
            elif random_uniform > 1 - probability:
                noised_image[i][j] = 255
    return noised_image
