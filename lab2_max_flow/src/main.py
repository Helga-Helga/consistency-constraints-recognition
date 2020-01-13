import os
import sys
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from noise import (
    add_laplacian_noise,
    add_gaussian_noise,
    add_salt_and_pepper_noise
)
from graph import *

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    if os.path.exists(image_path):
        input_image = mpimg.imread(image_path)
    else:
        raise Exception('Bad image path')
else:
    raise Exception('Usage: python main.py image_path')

print(input_image)

config = configparser.ConfigParser()
config.read('config.ini')
noise = config['NOISE']['noise']
loc_laplacian = float(config['LAPLACIAN_NOISE']['loc'])
scale_laplacian = float(config['LAPLACIAN_NOISE']['scale'])
loc_gaussian = float(config['GAUSSIAN_NOISE']['loc'])
scale_gaussian = float(config['GAUSSIAN_NOISE']['scale'])
prob_salt_and_pepper = float(config['SALT_AND_PEPPER_NOISE']['probability'])
L = float(config['EDGE_WEIGHT']['L'])
S = float(config['EDGE_WEIGHT']['S'])
beta = float(config['NODE_WEIGHT']['beta'])
number_of_iterations = int(config['ALGORITHM']['number_of_iterations'])

fig = plt.figure()
spec = fig.add_gridspec(ncols=3, nrows=1)

ax_original_image = fig.add_subplot(spec[0, 0])
ax_original_image.set_title('Original image')
ax_original_image.imshow(input_image, cmap=plt.get_cmap('gray'))

ax_noised_image = fig.add_subplot(spec[0, 1])
ax_noised_image.set_title('Noised image')

print(noise)
if noise == "L":
    noised_image = add_laplacian_noise(input_image, loc_laplacian, scale_laplacian)
elif noise == "G":
    noised_image = add_gaussian_noise(input_image, loc_gaussian, scale_gaussian)
elif noise == "SP":
    noised_image = add_salt_and_pepper_noise(input_image, prob_salt_and_pepper)
else:
    raise ValueError("Unknown noise")

ax_noised_image.imshow(noised_image, cmap=plt.get_cmap('gray'))

maxflow_graph = MaxFlowGraph(L, S, beta, noised_image)
maxflow_graph.alpha_expansion(number_of_iterations)
resulting_image = maxflow_graph.image

ax_result = fig.add_subplot(spec[0, 2])
ax_result.set_title('Result')
ax_result.imshow(resulting_image, cmap=plt.get_cmap('gray'))

for ax in [ax_original_image, ax_noised_image, ax_result]:
    ax.set_axis_off()

plt.show()
