import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from utils import rgb2gray
from noise import add_laplacian_noise

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    if os.path.exists(image_path):
        input_image = rgb2gray(mpimg.imread(image_path))
    else:
        raise Exception('Bad image path')
else:
    raise Exception('Usage: python main.py image_path')


fig = plt.figure()
spec = fig.add_gridspec(ncols=2, nrows=1)

ax_original_image = fig.add_subplot(spec[0, 0])
ax_original_image.imshow(input_image, cmap=plt.get_cmap('gray'))

ax_laplacian = fig.add_subplot(spec[0, 1])
ax_laplacian.imshow(add_laplacian_noise(input_image), cmap=plt.get_cmap('gray'))

for ax in [ax_original_image, ax_laplacian]:
    ax.set_axis_off()

plt.show()
