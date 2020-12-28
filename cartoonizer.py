import copy
import os

from skimage import feature
from skimage import exposure

from skimage import io

import numpy as np
import matplotlib.pyplot as plt


def bilateral_filter(img_in, sigma_s, sigma_v):
    """
    Bilateral Filtering of an input image

    Args:
        img_in: input image
        sigma_s: gaussian std. dev (spatial).
        sigma_v: gaussian std. dev.
    Returns:
        result: output image
    """
    # Gaussian function
    def gaussian(r2, sigma):
        return np.exp(-0.5 * r2 / sigma**2)

    # window width
    window_size = int(3 * sigma_s + 1)

    # initialize the results and sum of weights to very small values
    wgt_sum = np.zeros_like(img_in)
    result = img_in * (1e-7)

    # calculate the weights and accumulate the weight sum and result image
    for x in range(-window_size, window_size + 1):
        for y in range(-window_size, window_size + 1):
            # compute the spatial weight
            w = gaussian(x**2 + y**2, sigma_s)

            # shift by the offsets
            off = np.roll(img_in, [y, x], axis=[0, 1])

            # compute the value weight
            tw = w * gaussian((off - img_in)**2, sigma_v)

            # accumulate the results
            result += off * tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum


def cartoonizer_id(image):
    '''
    Cartoonize the image

    Args:
        image: input image to be cartoonized
    Returns:
        None
    '''
    # original image
    original_image = copy.deepcopy(image)

    # create empty edge image
    edge_image = np.zeros_like(image)

    # get edge map on each channel of image
    canny_r_image = feature.canny(image[:, :, 0], sigma=1.5)
    canny_g_image = feature.canny(image[:, :, 1], sigma=1.5)
    canny_b_image = feature.canny(image[:, :, 2], sigma=1.5)

    # get input image pixel value for edge pixels to form an image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if canny_r_image[i, j]:
                edge_image[i, j, 0] = image[i, j, 0]
            if canny_g_image[i, j]:
                edge_image[i, j, 1] = image[i, j, 1]
            if canny_b_image[i, j]:
                edge_image[i, j, 2] = image[i, j, 2]

    # edge contrasting
    edge_image = exposure.adjust_gamma(edge_image, 1.5)

    # sharpen the edges where edge is found
    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            if edge_image[i, j, 0] > 0:
                image[i, j, 0] = image[i, j, 0] * 0.7
            if edge_image[i, j, 1] > 0:
                image[i, j, 1] = image[i, j, 1] * 0.7
            if edge_image[i, j, 2] > 0:
                image[i, j, 2] = image[i, j, 2] * 0.7

    # run modified bilateral filter
    cartoon_image = np.stack([
        bilateral_filter(image[:, :, 0], 6.0, 0.1),
        bilateral_filter(image[:, :, 1], 6.0, 0.1),
        bilateral_filter(image[:, :, 2], 6.0, 0.1)], axis=2)

    # plots
    fig, axes = plt.subplots(1, ncols=2, figsize=(16, 8))
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    axes[1].imshow(cartoon_image)
    axes[1].axis('off')
    axes[1].set_title('Cartoonized Image')

    # save plot
    plt.savefig('output_cartoonizer.jpg')
    plt.show()


if __name__ == '__main__':
    # get image name from user
    input_img = input('Enter input image name: ')
    
    # input image
    image = io.imread(os.path.join(
            'input_images',
            input_img)
        ).astype(np.float32)/255.0
    # run cartoonizer
    cartoonizer_id(image)
