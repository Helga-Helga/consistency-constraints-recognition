from numpy import dot


def rgb2gray(image):
    """Convertion of 3D RGB image to 2D greyscale

    Parameters
    ----------
    image : numpy 2D array
        Input image

    Returns
    -------
    numpy 2D array
        Greyscale image
    """
    if len(image.shape) > 2:
        return dot(
            image[..., :3], [0.2989, 0.5870, 0.1140]
        ).round().astype(int)
    else:
        return image


def edge_weight(label1, label2, L, S):
    """Computing of edge weight between two labels for initial problem

    Parameters
    ----------
    label1: int
        Intensity of one object (pixel)
    label2: int
        Intensity of other object
    L: float
        Scale factor
    S: float
        Parameter similar to deviation

    Returns
    -------
    float
        Edge weight
    """
    return L * log(1 + (label1 - label2).pow(2) / (2 * S.pow(2)))


def node_weight(label, noised_color):
    """Computing of node weight for the given label in object

    Parameters
    ----------
    label: int
        Intensity of the pixel in initial image
    noised_color: int
        Intensity of the pixel in noised image

    Returns
    -------
    int
        Node weight
    """
    return (label - noised_color).pow(2)
