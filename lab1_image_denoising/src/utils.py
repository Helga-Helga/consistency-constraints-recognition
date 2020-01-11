from numpy import log


def neighbor_exists(image_height, image_width, i, j, neighbor_number):
    """Returns True if there is a given neighbor for a given pixel

    Parameters
    ----------
    image_height: unsigned integer
        Image height
    image_width: unsigned integer
        Image width
    i: unsigned integer
        Vertical coordinate of a pixel
    j:  unsigned integer
        Horizontal coordinate of a pixel
    neighbor_number: number from {0, 1, 2, 3}
        Neighbor index

    Returns
    -------
    True or False
        Returns True if neighbor exists, otherwise returns false
    """
    if neighbor_number == 0 and j > 0:
        return True
    elif neighbor_number == 1 and i > 0:
        return True
    elif neighbor_number == 2 and j < image_width - 1:
        return True
    elif neighbor_number == 3 and i < image_height - 1:
        return True
    else:
        return False


def get_neighbor_coordinate(i, j, neighbor_number):
    """Calculate coordinate of a given neighbor for a given pixel

    Parameters
    ----------
    i: unsigned integer
        Vertical coordinate of a pixel
    j:  unsigned integer
        Horizontal coordinate of a pixel
    neighbor_number: number from {0, 1, 2, 3}
        Neighbor index

    Returns
    -------
    tuple of unsigned integers
        Coordinated of neighbor
    """
    if neighbor_number == 0:
        return i, j - 1
    elif neighbor_number == 1:
        return i - 1, j
    elif neighbor_number == 2:
        return i, j + 1
    elif neighbor_number == 3:
        return i + 1, j
    else:
        return None, None


def edge_weight(label1, label2, beta):
    """Calculates edge weight

    Parameters
    ----------
    label1: binary value
        Label of one side of edge
    label2: binary value
        Label of another side of edge
    beta: number
        Edge weight if labels differ

    Returns
    -------
    number
        Edge weight
    """
    return 0 if (label1 == label2) else beta


def node_weight(label, noised_color, epsilon):
    """Calculates node weight

    Parameters
    ----------
    label: binary value
        Label assigned to the node
    noised_color: binary value
        Color of the node (pixel) in the noised image
    epsilon: number from [0, 1]
        Noised level
    """
    if label == noised_color:
        return -log(1 - epsilon)
    else:
        return -log(epsilon)
