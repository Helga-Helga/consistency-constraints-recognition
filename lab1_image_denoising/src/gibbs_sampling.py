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


def gibbs_sampling(initial_image, noised_image, epsilon, beta, iterations):
    print("Image denoising with Gibbs sampler...")
    height, width = initial_image.shape
    labeling = random.randint(2, size=(height, width))  # U{0, 1}
    sums_of_zero_labels = zeros(shape=(height, width))
    sums_of_unit_labels = zeros(shape=(height, width))
    for iteration in range(iterations):
        if iteration > save_after and iteration % save_step == 0:
            sums_of_zero_labels += labeling ^ 1
            sums_of_unit_labels += labeling
    imsave('images/labeling.png', labeling, cmap=gray)
    print("Resulting image is saved to \"images/labeling.png\"")
    return labeling


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    beta = float(config['EDGE_WEIGHT']['beta'])
    image_height = int(config['IMAGE_SIZE']['height'])
    image_width = int(config['IMAGE_SIZE']['width'])
    iterations_for_image_generation = int(
        config['ITERATIONS']['iterations_for_image'])
    initial_image = sample_input_image(
        image_height, image_width, beta, iterations_for_image_generation)
    epsilon = float(config['NOISE_LEVEL']['epsilon'])
    noised_image = add_noise(initial_image, epsilon)
    iterations_for_gibbs = int(config['ITERATIONS']['iterations_for_gibbs'])
    save_after = int(config['ITERATIONS']['save_after'])
    save_step = int(config['iterations']['save_step'])
    labeling = gibbs_sampling(initial_image, noised_image,
                              epsilon, beta,
                              iterations_for_gibbs, save_after, save_step)
