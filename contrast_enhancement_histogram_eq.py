from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

"""
This file contains an assignment I did for a class at university, namely Algorithms and Data Structures. 
The task was to enhance the contrast of an image, using histogram equalization.

All the functions work on greyscale images.
"""

#several functions used for the histogram equalization one 
def hist(img) -> list:
    """
    function for computing the histogram of image intensity values 
    returns the histogram (list)
    """
    img = np.asarray(img)
    hist1 = [0] * 256
    for row in img:
        for pixel in row:
            hist1[int(pixel)] += 1
    
    return hist1

def cdf(img) -> list:
    """
    function for computing teh cdf of an image
    returns the cdf (list)
    """
    hist1 = hist(img)
    pixels = img.shape[0] * img.shape[1]
    hist1 = [p/pixels for p in hist1]  #normalizing the values in the histogram
    cdf1 = np.cumsum(hist1)
    cdf1 = cdf1.tolist()

    return cdf1

def t_func(img) -> list:
    """
    T function, for normalizing the cdf
    returns the cdf, normalized (list)
    """
    cdf1 = cdf(img)
    nor = max(cdf1)
    T = [i / nor for i in cdf1]
    return T 

def histeq(inpimg, show_plots: bool) -> Image:
    """
    the histogram equalization function itself
    the user can choose to plot the before, after histogram and cdf
    returns the image, enhanced
    """
    inpimg = np.asarray(inpimg)
    outimg = np.empty((inpimg.shape[0], inpimg.shape[1]))
    T = t_func(inpimg)

    for i,row in enumerate(inpimg):
        for j,pixel in enumerate(row):   
            I = T[int(pixel)] * 255
            outimg[i,j] = I

    if show_plots:
        #plotting the before histogram
        hist0 = hist(inpimg)
        plt.plot(hist0, 'o')
        plt.show()

        #plotting the after histogram 
        hist1 = hist(outimg)
        plt.plot(hist1, 'o')
        plt.show()

        #plotting the before cdf 
        cdf0 = cdf(inpimg)
        plt.title('cdf before')
        plt.plot(cdf0)
        plt.show()

        #plotting the after cdf
        cdf1 = cdf(outimg)
        plt.title('cdf after')
        plt.plot(cdf1)
        plt.show()

    return Image.fromarray(outimg)

