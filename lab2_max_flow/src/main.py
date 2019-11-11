import os
import sys
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from utils import rgb2gray
from noise import (
    add_laplacian_noise,
    add_gaussian_noise,
    add_salt_and_pepper_noise
)

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    if os.path.exists(image_path):
        input_image = rgb2gray(mpimg.imread(image_path))
    else:
        raise Exception('Bad image path')
else:
    raise Exception('Usage: python main.py image_path')

config = configparser.ConfigParser()
config.read('config.ini')
loc_laplacian = float(config['LAPLACIAN_NOISE']['loc'])
scale_laplacian = float(config['LAPLACIAN_NOISE']['scale'])
loc_gaussian = float(config['GAUSSIAN_NOISE']['loc'])
scale_gaussian = float(config['GAUSSIAN_NOISE']['scale'])
prob_salt_and_pepper = float(config['SALT_AND_PEPPER_NOISE']['probability'])

fig = plt.figure()
spec = fig.add_gridspec(ncols=3, nrows=2)

ax_original_image = fig.add_subplot(spec[:-1, :])
ax_original_image.set_title('Original image')
ax_original_image.imshow(input_image, cmap=plt.get_cmap('gray'))

ax_laplacian = fig.add_subplot(spec[1, 0])
ax_laplacian.set_title('Laplacian noise')
ax_laplacian.imshow(
    add_laplacian_noise(input_image, loc_laplacian, scale_laplacian),
    cmap=plt.get_cmap('gray')
)

ax_gaussian = fig.add_subplot(spec[1, 1])
ax_gaussian.set_title('Gaussian noise')
ax_gaussian.imshow(
    add_gaussian_noise(input_image, loc_gaussian, scale_gaussian),
    cmap=plt.get_cmap('gray')
)

ax_salt_pepper = fig.add_subplot(spec[1, 2])
ax_salt_pepper.set_title('Salt-and-pepper noise')
ax_salt_pepper.imshow(
    add_salt_and_pepper_noise(input_image, prob_salt_and_pepper),
    cmap=plt.get_cmap('gray')
)

for ax in [ax_original_image, ax_laplacian, ax_gaussian, ax_salt_pepper]:
    ax.set_axis_off()

plt.show()
