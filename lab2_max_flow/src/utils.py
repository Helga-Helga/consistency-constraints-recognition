from numpy import dot

def rgb2gray(image):
    if len(image.shape) > 2:
        return dot(
            image[..., :3], [0.2989, 0.5870, 0.1140]
        ).round().astype(int)
    else:
        return image
