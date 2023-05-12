import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d

"""
This file contains my work on implementing several image processing functions, including rgb2gray conversion, image binarization,
canny edge detection etc.

Some of the functions only work on grayscale images due to the limitations of the ndimage.convolve function, 
which only accepts 2D arrays as input. However, treating each color channel individually would make applying the functions to non-grayscale
images possible.
"""

img = Image.open('tanis_half_elven.jpg')

def rgb2gray(image) -> Image:
    """
    rgb2gray conversion function 
    returns the greyscale of an image (Image)
    """
    image = np.asarray(image)

    return Image.fromarray(np.dot(image[...,:3], [0.299, 0.587, 0.114]))

def binarize_st(img, threshold) -> Image:
    """
    binarizer function, using single thresholding (works on grayscale images)
    returns the image, binarized (Image)
    """
    img = np.asarray(img)

    img[img < threshold] = 0
    img[img >= threshold] = 255

    return Image.fromarray(img)

def binarize_dt(img, threshold1, threshold2) -> Image:
    """
    binarizer function, using double thresholding (works on grayscale images)
    returns the image, binarized (Image)
    """
    img = np.asarray(img)

    img[img < threshold1] = 0
    img[img > threshold2] = 0
    img[(img >= threshold1) & (img <= threshold2)] = 255

    return Image.fromarray(img)

def gauss_blur(img) -> np.array:
    """
    gaussian blur function, which utilizes a 5x5 gaussian kernel 
    returns the image, blurred (array)
    """
    img = np.asarray(img)

    kernel = (1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]])

    return ndimage.convolve(img,  kernel, mode='constant', cval=0.0)


def img_gradient(img) -> np.array:
    """
    the function which calculates the gradient of the image intensity function,
    which utilizes a 3x3 sobel kernel (works on grayscale/binary images)
    returns the gradient matrix of an image (array)
    """
    img = np.asarray(img)

    x_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    
    y_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    
    x_grad = ndimage.convolve(img, x_kernel, mode='constant', cval=0.0)
    y_grad = ndimage.convolve(img, y_kernel, mode='constant', cval=0.0)

    grad = np.sqrt(np.square(x_grad) + np.square(y_grad)) #np.hypot     

    return grad, x_grad, y_grad

def grad_angles(x_grad, y_grad) -> np.array:
    """
    function, which computes the direction of the gradient at each pixel in an image 
    returns an array of angles, in radians 
    """
    return np.arctan2(x_grad, y_grad)

img = rgb2gray(img)
grad, x_grad, y_grad = img_gradient(img)
grad = Image.fromarray(grad)
grad.show()
angles = grad_angles(x_grad, y_grad)

def non_max_suppresion(grad, angles) -> np.array:
    """
    gradient thresholding function
    works on the gradient of an image
    returns the gradient, thresholded (array)
    """
    grad = np.asarray(grad)
    angles = np.asarray(angles)
    gmax = np.zeros(grad.shape)
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            if angles[i][j] < 0:
                angles[i][j] += 360

            if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
                # 0 degrees
                if (angles[i][j] >= 337.5 or angles[i][j] < 22.5) or (angles[i][j] >= 157.5 and angles[i][j] < 202.5):
                    if grad[i][j] >= grad[i][j + 1] and grad[i][j] >= grad[i][j - 1]:
                        gmax[i][j] = grad[i][j]
                # 45 degrees
                if (angles[i][j] >= 22.5 and angles[i][j] < 67.5) or (angles[i][j] >= 202.5 and angles[i][j] < 247.5):
                    if grad[i][j] >= grad[i - 1][j + 1] and grad[i][j] >= grad[i + 1][j - 1]:
                        gmax[i][j] = grad[i][j]
                # 90 degrees
                if (angles[i][j] >= 67.5 and angles[i][j] < 112.5) or (angles[i][j] >= 247.5 and angles[i][j] < 292.5):
                    if grad[i][j] >= grad[i - 1][j] and grad[i][j] >= grad[i + 1][j]:
                        gmax[i][j] = grad[i][j]
                # 135 degrees
                if (angles[i][j] >= 112.5 and angles[i][j] < 157.5) or (angles[i][j] >= 292.5 and angles[i][j] < 337.5):
                    if grad[i][j] >= grad[i - 1][j - 1] and grad[i][j] >= grad[i + 1][j + 1]:
                        gmax[i][j] = grad[i][j]
    return gmax

grad = non_max_suppresion(grad, angles)
mat = Image.fromarray(grad)
mat.show()

def adaptive_thresholding(grad) -> np.array:
    """
    function to threshold the gradient of an image, which utilizes adaptive thresholding 
    returns the gradient, thresholded (array)
    """
    grad_out = np.zeros(grad.shape)
    N, M = grad.shape
    NEIGHBOURS = 8
    COT = 10 #cot - Constant Of Thresholding

    for l in range(N):
        for k in range(M):
            mean = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (l + i >= 0 and l + i < N) and (k + j >= 0 and k + j < N): #check whether the index is out of range
                        mean += grad[l + i][k + j]
                        
            mean /= NEIGHBOURS
            thresh = mean - COT
            if grad[l,k] <= thresh:
                grad_out[l,k] = 0
            elif grad[l,k] >= thresh:
                grad_out[l,k] = 255
            
    return grad_out

grad_out = adaptive_thresholding(grad)
grad_out = Image.fromarray(grad_out)
grad_out.show()


