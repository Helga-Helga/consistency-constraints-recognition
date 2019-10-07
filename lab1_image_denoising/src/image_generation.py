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
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.rand() < epsilon:
                image[i][j] = 1 - image[i][j] % 2
    imsave('images/noised_image.png', image, cmap=gray)
    return image


def sample_input_image(height, width, beta, iterations):
    image = random.randint(2, size=(height, width))  # U([0, 1])
    for iteration in range(iterations):
        for i in range(height):
            for j in range(width):
                zero_weight = 0
                unit_weight = 0
                for n in range(4):
                    if neighbor_exists(height, width, i, j, n):
                        i_n, j_n = get_neighbor_coordinate(i, j, n)
                        zero_weight += edge_weight(0, image[i_n][j_n], beta)
                        unit_weight += edge_weight(1, image[i_n][j_n], beta)
                t = exp(-zero_weight) / (exp(-zero_weight) + exp(-unit_weight))
                x = random.rand()
                if x < t:
                    image[i][j] = 0
                else:
                    image[i][j] = 1
    imsave('images/input_image.png', image, cmap=gray)
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
