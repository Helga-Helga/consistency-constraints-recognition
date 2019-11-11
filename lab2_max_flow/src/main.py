import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import rgb2gray

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    if os.path.exists(image_path):
        input_image = rgb2gray(mpimg.imread(image_path))
    else:
        raise Exception('Bad image path')
else:
    raise Exception('Usage: python main.py image_path')


fig, ax = plt.subplots()
ax.imshow(input_image, cmap=plt.get_cmap('gray'))
plt.show()
