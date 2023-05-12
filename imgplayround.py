import numpy as np
from PIL import Image
from scipy import ndimage

"""
This file contains my work on implementing several image processing functions, including rgb2gray conversion, image binarization,
canny edge detection etc.

Some of the functions only work on grayscale images due to the limitations of the ndimage.convolve function, 
which only accepts 2D arrays as input. However, treating each color channel individually would make applying the functions to non-grayscale
images possible.


"""

#TODO: vectorize the nested  loops
#TODO: tweak NEIGHBOURS, COT depending on the stats of an image
#TODO: try different edge-handling techniques

def rgb2gray(image) -> Image:
    """
    rgb2gray conversion function 
    accepts both arrays and images (PIL, Image)
    returns the greyscale of an image (PIL, Image)
    """
    image = np.asarray(image)

    return Image.fromarray(np.dot(image[...,:3], [0.299, 0.587, 0.114])) #grayscale = 0.299 R + 0.587 G + 0.114 B 

def binarize_st(img, threshold) -> Image:
    """
    binarizer function, using single thresholding (works on grayscale images)
    accepts both arrays and images (PIL, Image) 
    returns the image, binarized (PIL, Image)
    """
    img = np.asarray(img)

    img[img < threshold] = 0
    img[img >= threshold] = 255

    return Image.fromarray(img)

def binarize_dt(img, threshold1, threshold2) -> Image:
    """
    binarizer function, using double thresholding (works on grayscale images)\n
    accepts both arrays and images (PIL, Image)\n
    returns the image, binarized (PIL, Image)
    """
    img = np.asarray(img)

    img[img < threshold1] = 0
    img[img > threshold2] = 0
    img[(img >= threshold1) & (img <= threshold2)] = 255

    return Image.fromarray(img)

def gauss_blur(img) -> np.ndarray:
    """
    gaussian blur function, which utilizes a 5x5 gaussian kernel\n
    accepts both arrays and images (PIL, Image)\n
    returns the image, blurred (array)
    """
    img = np.asarray(img)

    #gaussian blur kernel
    kernel = (1 / 159.0) * np.array([[2, 4, 5, 4, 2],
                                   [4, 9, 12, 9, 4],
                                   [5, 12, 15, 12, 5],
                                   [4, 9, 12, 9, 4],
                                   [2, 4, 5, 4, 2]]) 

    return ndimage.convolve(img,  kernel, mode='constant', cval=0.0)

def img_gradient(img) -> np.ndarray:
    """
    the function which calculates the gradient of the image intensity function,
    which utilizes a 3x3 sobel kernel (works on grayscale/binary images)\n
    accepts both arrays and images (PIL, Image)\n
    returns the gradient matrix of an image (array)
    """
    img = np.asarray(img)

    #x gradient kernel
    x_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    
    #y gradient kernel
    y_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    
    x_grad = ndimage.convolve(img, x_kernel, mode='constant', cval=0.0)
    y_grad = ndimage.convolve(img, y_kernel, mode='constant', cval=0.0)

    #same as np.hypot; performs the Pythagorean addition operation on the entries of the two matrices to obtain the full gradient matrix
    grad = np.sqrt(np.square(x_grad) + np.square(y_grad))  

    return grad, x_grad, y_grad

def grad_angles(x_grad: np.ndarray, y_grad:np.ndarray) -> np.ndarray:
    """
    function, which computes the direction of the gradient at each pixel in an image\n 
    :x_grad: np.ndarray\n
    :y_grad: np.ndarray\n
    where x_grad, y_grad are the x and y gradients of the image, respectively\n
    returns an array of angles, in radians 
    """
    return np.arctan2(x_grad, y_grad) #computing the angle in between the values, in radians

def non_max_suppresion(grad: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    gradient thresholding function\n
    :grad: np.ndarray\n
    :angles: np.ndarray\n
    where grad is the gradient of an image, and angles is the direction of the gradient at each pixel\n
    returns the gradient, thresholded (array)
    """
    grad = np.asarray(grad)
    angles = np.asarray(angles)
    gmax = np.zeros(grad.shape)

    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            if angles[i][j] < 0:
                angles[i][j] += 360

            #if statements, which round up the angle, depending on the interval it happens to be in  
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

def adaptive_thresholding(grad: np.ndarray) -> np.ndarray:
    """
    function to threshold the gradient of an image, which utilizes adaptive thresholding with the mean value filter\n
    :grad: np.ndarray\n
    where grad is the gradient of an image\n
    returns the gradient, thresholded (array)
    """
    grad_out = np.zeros(grad.shape)
    N = grad.shape[0]
    M = grad.shape[1]
    NEIGHBOURS = 8 
    COT = 15 #cot - Constant Of Thresholding

    for l in range(N):
        for k in range(M):
            mean = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (l + i >= 0 and l + i < N) and (k + j >= 0 and k + j < M): #check whether the indices (l+i; k+j) is out of range
                        mean += grad[l + i][k + j]

            mean -= grad[l,k] #subtract the current cell, since it was counted in as well
            mean /= NEIGHBOURS 
            thresh = mean - COT 

            if grad[l, k] >= thresh:
                grad_out[l, k] = 0
            else:
                grad_out[l, k] = 255
            
    return grad_out

def hysteresis_filter(edges: np.ndarray) -> np.ndarray:
    """
    function to filter the previously found edges by hysteresis\n
    :edges: np.ndarray\n
    returns the edges, filtered (array)
    """ 
    edges = np.asarray(edges)
    edges_out = np.zeros(edges.shape)
    N = edges.shape[0]
    M = edges.shape[1]

    #counting the total number of white pixels around each pixel
    for l in range(N):
        for k in range(M):
            neighbours = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (l + i >= 0 and l + i < N) and (k + j >= 0 and k + j < M): 
                      neighbours += edges[l+i][k+j] 

            if neighbours == 0:
                edges_out[l][k] = 0
            else:
                edges_out[l][k] = 255

    return edges_out

def canny_detection(img, show: bool) -> np.ndarray:
    """
    function, which detects the edges in an image, which utilizes the Canny edge detection algorithm\n
    the user can choose, whether to show the result\n
    acceps either np.ndarray of PIL, Image\n
    returns the edges, extracted from the image (array)
    """
    img = rgb2gray(img)
    img = gauss_blur(img)
    grad, x_grad, y_grad = img_gradient(img)
    angles = grad_angles(x_grad, y_grad)
    gmax = non_max_suppresion(grad, angles)
    edges = adaptive_thresholding(grad)
    edges = hysteresis_filter(edges)

    if show:
        edges = Image.fromarray(edges)
        edges.show()









