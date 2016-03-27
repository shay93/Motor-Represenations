import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

#first read input image into an array

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

image_str = 'grey_cat.jpg'
img = rgb2gray(plt.imread(image_str))
input_data = img

#set output equal to the input data
output = input_data


#scale input data to in between 0 and 1 
input_data = np.divide(input_data,255)
output_data = np.divide(output,255)
