from numpy import log


def neighbor_exists(image_height, image_width, i, j, neighbor_number):
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
    if label1 == label2:
        return 0
    else:
        return beta


def node_weight(initial_color, noised_color, epsilon):
    if initial_color == noised_color:
        return -log(1 - epsilon)
    else:
        return -log(epsilon)
