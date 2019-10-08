from numpy import (
    random,
    exp,
    zeros,
)
from matplotlib.pyplot import imsave
from matplotlib.cm import gray

from utils import (
    node_weight,
    edge_weight,
    neighbor_exists,
    get_neighbor_coordinate,
)
from image_generation import (
    sample_input_image,
    add_noise,
)


def get_labeling(sum_of_zero_labels, sum_of_unit_labels):
    height, width = sum_of_zero_labels.shape
    labeling = zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            if sum_of_zero_labels[i][j] > sum_of_unit_labels[i][j]:
                labeling[i][j] = 0
            else:
                labeling[i][j] = 1
    return labeling


def gibbs_sampling(initial_image, noised_image, epsilon, beta, iterations):
    print("Image denoising with Gibbs sampler...")
    height, width = initial_image.shape
    labeling = random.randint(2, size=(height, width))
    sum_of_zero_labels = zeros(shape=(height, width))
    sum_of_unit_labels = zeros(shape=(height, width))
    for iteration in range(iterations):
        for i in range(height):
            for j in range(width):
                for n in range(4):
                    sum_zero_edges = 0
                    sum_unit_edges = 0
                    if neighbor_exists(height, width, i, j, n):
                        i_n, j_n = get_neighbor_coordinate(i, j, n)
                        sum_zero_edges += edge_weight(
                            0, labeling[i_n][j_n], beta)
                        sum_unit_edges += edge_weight(
                            1, labeling[i_n][j_n], beta)
                zero = exp(-node_weight(
                    0, noised_image[i][j], epsilon) - sum_zero_edges)
                unit = exp(-node_weight(
                    1, noised_image[i][j], epsilon) - sum_zero_edges)
                t = zero / (zero + unit)
                x = random.rand()
                if x < t:
                    labeling[i][j] = 0
                else:
                    labeling[i][j] = 1
                if iteration > 2000 and iteration % 30 == 0:
                    if labeling[i][j] == 0:
                        sum_of_zero_labels += 1
                    else:
                        sum_of_unit_labels += 1
    result = get_labeling(sum_of_zero_labels, sum_of_unit_labels)
    imsave('images/labeling.png', labeling, cmap=gray)
    print("Resulting image is saved to \"images/labeling.png\"")
    return labeling


if __name__ == "__main__":
    beta = 1.0
    initial_image = sample_input_image(20, 20, beta, 5000)
    epsilon = 0.1
    noised_image = add_noise(initial_image, epsilon)
    labeling = gibbs_sampling(
        initial_image, initial_image, epsilon, beta, 10000)
