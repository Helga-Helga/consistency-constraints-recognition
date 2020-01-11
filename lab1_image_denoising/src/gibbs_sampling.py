import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import maxflow

from numpy import (
    random,
    exp,
    zeros,
    int_,
    logical_not,
)

from image_generation import (
    sample_input_image,
    add_noise,
)
from utils import (
    node_weight,
    edge_weight,
    neighbor_exists,
    get_neighbor_coordinate,
)


def get_labeling(sums_of_zero_labels, sums_of_unit_labels):
    """Returns image of the most common colors occured in the labeling

    Parameters
    ----------
    sums_of_zero_labels: matrix of unsigned int values
        Matrix with image shape where cell (i, j)
        contains how many times this cell had value 0 in labeling
        during iterations of Gibbs sampler
    sums_of_unit_labels: matrix of unsigned int values
        Matrix with image shape where cell (i, j)
        contains how many times this cell had value 1 in labeling
        during iterations of Gibbs sampler

    Returns
    -------
    matrix of binary values
        Returns image of the most common colors occured in the labeling
        during Gibbs sampler iterations
    """
    height, width = sums_of_zero_labels.shape
    labeling = zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            labeling[i, j] = int(
                sums_of_zero_labels[i, j] <= sums_of_unit_labels[i, j])
    return labeling


def gibbs_iteration(labeling, height, width, epsilon, beta):
    """One iteration of Gibbs sampler

    Parameters
    ----------
    labeling: matrix of image size with binary values
        Current labeling
    height: unsigned int
        Image height
    widht: unsigned int
        Image width
    epsilon: number
        Noise level
    beta: number
        Weight of edge if its labels differ

    Returns
    -------
    matrix of image size with binary values
        Updated labeling
    """
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
            labeling[i, j] = int(random.uniform() >= t)  # U[0, 1]
    return labeling


def almost_equal_labelings(labeling1, labeling2, changes_threshold):
    """Returns True if there are not more than
    changes_threshold% of mismatching pixels

    Parameters
    ----------
    labeling1: matrix of binary values of image size
        Labeling
    labeling2: matrix of binary values of image size
        Labeling from neighbor iteration of Gibbs sampler
    changes_threshold: number
        Percent of maximum number of mismatching pixels to consider
        the givel labelings almost equal

    Returns
    -------
    True or False
        Are the given labelings almost equal
    """
    height, width = labeling1.shape
    max_errors = height * width * changes_threshold / 100
    current_errors = 0
    for i in range(height):
        for j in range(width):
            if labeling1[i, j] != labeling2[i, j]:
                current_errors += 1
            if current_errors > max_errors:
                return False
    return True


def gibbs_sampling(initial_image, noised_image,
                   epsilon, beta,
                   changes_threshold):
    """Gibbs sampling algorithm implementation

    Parameters
    ----------
    initial_image: matrix of binary values
        Generated image
    noised_image: matrix of binary values
        Generated image after applying noise
    epsilon: number
        Noise level
    beta: number
        Weight of edge if its labels differ
    changes_threshold: number
        Percent of maximum number of mismatching pixels to consider
        the givel labelings almost equal

    Returns
    -------
    matrix of binary values
        Labeling as a reconstructed image
    """
    print("Image denoising with Gibbs sampler...")
    height, width = initial_image.shape
    labeling = random.randint(2, size=(height, width))  # U{0, 1}
    sums_of_zero_labels = zeros(shape=(height, width))
    sums_of_unit_labels = zeros(shape=(height, width))
    iteration = 0
    while True:
        iteration += 1
        labeling_prev = labeling.copy()
        labeling = gibbs_iteration(labeling, height, width, epsilon, beta)
        if iteration > 5 and iteration % 2 == 0:
            # Save labeling
            sums_of_zero_labels += labeling ^ 1
            sums_of_unit_labels += labeling
        # Break iterations when not more than 5% of pixels have changed
        if almost_equal_labelings(
                labeling_prev, labeling, changes_threshold):
            break
        print("Iteration # {}".format(iteration))
    result = get_labeling(sums_of_zero_labels, sums_of_unit_labels)
    return labeling


def count_errors(image, labeling):
    """Counts number of mismatching pixels in the initial image and labeling

    Parameters
    ----------
    image: matrix of binary values
        Initial (generated) image
    labeling: matrix of binary values
        Reconstructed image using Gibbs sampler

    Returns
    -------
    integer
        Number of incorrectly recognized pixels
    """
    errors = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] != labeling[i, j]:
                errors += 1
    percent = 100 * errors / (image.shape[0] * image.shape[1])
    print("Number of incorrectly recognized pixels: {}, it is {}% of image"
          .format(errors, percent))


def maxflow_image_restoration(noised_image):
    """Simple maxflow binary image restoration

    Parameters
    ----------
    noised_image: matrix of binary values
        Generated image after noising

    Returns
    -------
    matrix of binary values
        Restored image
    """
    # Create the graph
    g = maxflow.Graph[int]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid
    nodeids = g.add_grid_nodes(noised_image.shape)
    # Add non-terminal edges with the same capacity
    g.add_grid_edges(nodeids, 1)
    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node
    g.add_grid_tedges(nodeids, noised_image, 1 - noised_image)
    # Find the maximum flow
    g.maxflow()
    # Get the segments of the nodes in the grid
    sgm = g.get_grid_segments(nodeids)
    # The labels should be 1 where sgm is False and 0 otherwise
    resulting_image = int_(logical_not(sgm))
    return resulting_image


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

    changes_threshold = int(config['ITERATIONS']['changes_threshold'])
    labeling = gibbs_sampling(initial_image, noised_image,
                              epsilon, beta_gibbs,
                              changes_threshold)
    count_errors(initial_image, labeling)

    maxflow_result = maxflow_image_restoration(noised_image)

    fig = plt.figure()
    spec = fig.add_gridspec(ncols=4, nrows=1)

    ax_original_image = fig.add_subplot(spec[0, 0])
    ax_original_image.set_title('Generated image')
    ax_original_image.imshow(initial_image, cmap=plt.get_cmap('gray'))

    ax_noised_image = fig.add_subplot(spec[0, 1])
    ax_noised_image.set_title('Noised image')
    ax_noised_image.imshow(noised_image, cmap=plt.get_cmap('gray'))

    ax_gibbs = fig.add_subplot(spec[0, 2])
    ax_gibbs.set_title('After Gibbs sampling')
    ax_gibbs.imshow(labeling, cmap=plt.get_cmap('gray'))

    ax_maxflow = fig.add_subplot(spec[0, 3])
    ax_maxflow.set_title('After maxflow')
    ax_maxflow.imshow(maxflow_result, cmap=plt.get_cmap('gray'))

    for ax in [ax_original_image, ax_noised_image, ax_gibbs, ax_maxflow]:
        ax.set_axis_off()

    plt.show()
