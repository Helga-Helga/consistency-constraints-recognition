import argparse
from numpy import (
    random,
    exp,
)
from matplotlib.pyplot import imsave
from matplotlib.cm import gray

from utils import (
    neighbor_exists,
    get_neighbor_coordinate,
    edge_weight,
)


def add_noise(image, epsilon):
    """Adding noise to image by inverting pixels

    Parameters
    ----------
    image: matrix of binary values
        Initial image
    epsilon: number from [0, 1]
        Probability of inverting a pixel

    Returns
    -------
    matrix of binary values
        Noised image
    """
    print("Adding noise with epsilon =", epsilon)
    noised_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.uniform() < epsilon:  # U[0, 1]
                noised_image[i, j] = 1 - image[i, j]
    imsave('images/noised_image.png', noised_image, cmap=gray)
    print("Noised image is saved to \"images/noised_image.png\"")
    return noised_image


def sample_input_image(height, width, beta, iterations):
    """Generation of image using Gibbs sampler

    Parameters
    ----------
    height: unsigned integer
        Image height
    widht: unsigned integer
        Image width
    beta: number
        Weight of edge if its labels differ
    iterations: unsigned integer
        Number of iterations of image generation

    Retunrs
    -------
    matrix of binary values of size (height, width)
        Generated binary image
    """
    print("Generating input image", height, "x", width)
    image = random.randint(2, size=(height, width))  # U{0, 1}
    for iteration in range(iterations):
        for i in range(height):
            for j in range(width):
                zero_weight = 0  # sum of edges weights going from zero label
                unit_weight = 0  # sum of edges weights going from unit label
                for n in range(4):  # for all neighbors
                    if neighbor_exists(height, width, i, j, n):
                        i_n, j_n = get_neighbor_coordinate(i, j, n)
                        zero_weight += edge_weight(0, image[i_n, j_n], beta)
                        unit_weight += edge_weight(1, image[i_n, j_n], beta)
                t = exp(-zero_weight) / (exp(-zero_weight) + exp(-unit_weight))
                image[i, j] = int(random.uniform() >= t)  # U [0, 1]
    imsave('images/input_image.png', image, cmap=gray)
    print("Generated image is saved to \"images/input_image.png\"")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate image')
    parser.add_argument("image_height", type=int,
                        help='image height')
    parser.add_argument("image_width", type=int,
                        help='image width')
    parser.add_argument("edge_weight", type=float,
                        help="edge weight if nodes labels are different")
    parser.add_argument("epsilon", type=float,
                        help="probability of color change for noising")
    args = parser.parse_args()

    image = sample_input_image(
        args.image_height, args.image_width, args.edge_weight, 5000)
    add_noise(image, args.epsilon)
