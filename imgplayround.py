import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d

"""
This file contains my work on implementing several image processing functions, including rgb2gray conversion, image binarization,
canny edge detection etc.

Some of the functions only work on grayscale images due to the limitations of the ndimage.convolve function, 
which only accepts 2D arrays as input.
"""

img = Image.open('pumpkinme.jpg')

def rgb2gray(image):
    """
    rgb2gray conversion function 
    """
    image = np.asarray(image)

    return Image.fromarray(np.dot(image[...,:3], [0.299, 0.587, 0.114]))

def binarize_st(img, threshold):
    """
    binarizer function, using single thresholding (works on grayscale images)
    """
    img = np.asarray(img)

    img[img < threshold] = 0
    img[img >= threshold] = 255

    return Image.fromarray(img)

def binarize_dt(img, threshold1, threshold2):
    """
    binarizer function, using double thresholding (works on grayscale images)
    """
    img = np.asarray(img)

    img[img < threshold1] = 0
    img[img > threshold2] = 0
    img[(img >= threshold1) & (img <= threshold2)] = 255

    return Image.fromarray(img)

def gauss_blur(img):
    """
    gaussian blur function, which utilizes a 5x5 gaussian kernel 
    returns the image, blurred
    """
    img = np.asarray(img)
    kernel = (1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]])

    return ndimage.convolve(img,  kernel, mode='constant', cval=0.0)


def img_gradient(img):
    """
    the function which calculates the gradient of the image intensity function,
    which utilizes a 3x3 sobel kernel (works on grayscale images)
    """
    x_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    
    y_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    
    x_grad = ndimage.convolve(img, x_kernel, mode='constant', cval=0.0)
    y_grad = ndimage.convolve(img, y_kernel, mode='constant', cval=0.0)

    grad = np.sqrt(np.square(x_grad) + np.square(y_grad))

    return grad

grad = img_gradient(img)
Image.fromarray(grad).show()

