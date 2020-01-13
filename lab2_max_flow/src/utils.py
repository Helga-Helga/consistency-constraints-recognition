from numpy import (
    dot,
    zeros,
    reshape,
    where,
    array,
    clip,
)

lookup_table = zeros((256, 256))
for i in range(256):
    for j in range(256):
        lookup_table[i, j] = (i - j) ** 2


def neighbor_exists(i, j, neighbor_index, height, width):
    """Returns True if a given neighbor exists for a given pixel

    Parameters
    ----------
    i: unsigned integer
        Vertical coordinate of a pixel
    j: unsigned integer
        Horizontal coordinate of a pixel
    neighbor_index: belongs to {0, 1, 2, 3}
        Index of a neighbor
    height: unsigned integer
        Image height
    width: unsigned integer
        Image width

    Returns
    -------
    True of False
        Result depends on existence of a given neighbor
    """
    if neighbor_index == 0 and j > 0:
        return True
    elif neighbor_index == 1 and i > 0:
        return True
    elif neighbor_index == 2 and j + 1 < width:
        return True
    elif neighbor_index == 3 and i + 1 < height:
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
