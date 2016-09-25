from __future__ import division 
import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.pyplot as plt
import png
import pickle


def save_normalized_weights(path_to_conv1,path_to_conv2):
	"""
	Takes an input of weights and saves them as images so that training may be observed
	inputs: A sessions object to evaluate the weights
	"""
	W_conv1 = pickle.load(open(path_to_conv1,'rb'))
	W_conv2 = pickle.load(open(path_to_conv2,'rb'))

	#initialize a figure to store the images
	conv1_fig = plt.figure(1,(20.,20.))
	#initialize an image grid
	conv1_grid = ImageGrid(conv1_fig, 111,nrows_ncols=(32 // 8,8),axes_pad = 0.1) 
	for i in range(32):
		kernel = W_conv1[:,:,0,i]
		kernel_normed = np.divide(kernel,np.mean(kernel)) * 255
		conv1_grid[i].imshow(kernel_normed, cmap = "Greys_r")
	
	conv1_fig.savefig("Image_Autoencoder_Ver2_Outputs/Conv1_Kernels.png")
	plt.close(conv1_fig)
	#perform the above for second conv layer as well
	conv2_fig = plt.figure(1,(20.,20.))
	conv2_grid = ImageGrid(conv2_fig, 111,nrows_ncols=(64 // 8,8),axes_pad = 0.1)
	for j in range(64):
		kernel = W_conv2[:,:,0,j]
		kernel_normed = np.divide(kernel,np.mean(kernel)) * 255
		conv2_grid[j].imshow(kernel_normed,cmap = "Greys_r")

	conv2_fig.savefig("Image_Autoencoder_Ver2_Outputs/Conv2_Kernels.png")
	plt.close(conv2_fig)


save_normalized_weights("Image_Autoencoder_Ver2_Outputs/W_conv1.npy","Image_Autoencoder_Ver2_Outputs/W_conv2.npy")

loss = pickle.load(open("Image_Autoencoder_Ver2_Outputs/loss.npy",'rb'))
f = plt.figure()
plt.plot(loss)
plt.title('loss')
f.savefig("Image_Autoencoder_Ver2_Outputs/loss.png")