import configparser

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


def get_labeling(sums_of_zero_labels, sums_of_unit_labels):
    height, width = sums_of_zero_labels.shape
    labeling = zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            if sums_of_zero_labels[i][j] > sums_of_unit_labels[i][j]:
                labeling[i][j] = 0
            else:
                labeling[i][j] = 1
    return labeling


def gibbs_iteration(labeling, height, width, iteration, epsilon):
    for i in range(height):
        for j in range(width):
            sum_zero_edges = 0
            sum_unit_edges = 0
            for n in range(4):
                if neighbor_exists(height, width, i, j, n):
                    i_n, j_n = get_neighbor_coordinate(i, j, n)
                    sum_zero_edges += edge_weight(
                        0, noised_image[i_n, j_n], beta)
                    sum_unit_edges += edge_weight(
                        1, noised_image[i_n, j_n], beta)
            zero = exp(-node_weight(
                0, noised_image[i, j], epsilon) - sum_zero_edges)
            unit = exp(-node_weight(
                1, noised_image[i, j], epsilon) - sum_unit_edges)
            t = zero / (zero + unit)
            labeling[i][j] = int(random.uniform() >= t)  # U[0, 1]
    return labeling


def almost_equal_labelings(labeling1, labeling2, error_rate):
    height, width = labeling1.shape
    max_errors = height * width * error_rate / 100
    current_errors = 0
    for i in range(height):
        for j in range(width):
            if labeling1[i][j] != labeling2[i][j]:
                current_errors += 1
            if current_errors > max_errors:
                return False
    return True


def gibbs_sampling(initial_image, noised_image,
                   epsilon, beta):
    print("Image denoising with Gibbs sampler...")
    height, width = initial_image.shape
    labeling = random.randint(2, size=(height, width))  # U{0, 1}
    sums_of_zero_labels = zeros(shape=(height, width))
    sums_of_unit_labels = zeros(shape=(height, width))
    for iteration in range(iterations):
        labeling_prev = labeling.copy()
        labeling = gibbs_iteration(labeling, height, width, iteration, epsilon)
        if iteration > save_after and iteration % save_step == 0:
            sums_of_zero_labels += labeling ^ 1
            sums_of_unit_labels += labeling
        if almost_equal_labelings(labeling_prev, labeling, 5):
            break
        print(iteration)
    result = get_labeling(sums_of_zero_labels, sums_of_unit_labels)
    imsave('images/labeling.png', labeling, cmap=gray)
    print("Resulting image is saved to \"images/labeling.png\"")
    return labeling


def count_errors(image, noised_image, labeling):
    errors = 0
    noised_errors = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] != labeling[i][j]:
                errors += 1
            if noised_image[i][j] != labeling[i][j]:
                noised_errors += 1
    percent = 100 * errors / (image.shape[0] * image.shape[1])
    print("Number of incorrectly recognized pixels: {}, it is {}% of image"
          .format(errors, percent))
    print("Number of errors from noised image: {}".format(noised_errors))


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    beta_image = float(config['EDGE_WEIGHT']['beta_image'])
    beta_gibbs = float(config['EDGE_WEIGHT']['beta_gibbs'])
    image_height = int(config['IMAGE_SIZE']['height'])
    image_width = int(config['IMAGE_SIZE']['width'])
    iterations_for_image_generation = int(
        config['ITERATIONS']['iterations_for_image'])

    initial_image = sample_input_image(
        image_height, image_width, beta_image, iterations_for_image_generation)

    epsilon = float(config['NOISE_LEVEL']['epsilon'])
    noised_image = add_noise(initial_image, epsilon)

    labeling = gibbs_sampling(initial_image, noised_image, epsilon, beta_gibbs)
